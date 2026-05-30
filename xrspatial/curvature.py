from __future__ import annotations

# std lib
from functools import partial
from typing import Optional, Union

# 3rd-party
try:
    import cupy
except ImportError:
    class cupy(object):
        ndarray = False

try:
    import dask.array as da
except ImportError:
    da = None

import numpy as np
import xarray as xr
from numba import cuda

# local modules
from xrspatial.utils import ArrayTypeFunctionMapping
from xrspatial.utils import _boundary_to_dask
from xrspatial.utils import _pad_array
from xrspatial.utils import _validate_boundary
from xrspatial.utils import _validate_raster
from xrspatial.utils import cuda_args
from xrspatial.utils import get_dataarray_resolution
from xrspatial.utils import ngjit
from xrspatial.dataset_support import supports_dataset


@ngjit
def _cpu(data, cellsize_x, cellsize_y):
    out = np.empty(data.shape, np.float32)
    out[:] = np.nan
    rows, cols = data.shape
    csx = np.float64(cellsize_x)
    csy = np.float64(cellsize_y)
    csx2 = csx * csx
    csy2 = csy * csy
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            d = np.float64(data[y + 1, x]) + np.float64(data[y - 1, x])
            d = d / np.float64(2) - np.float64(data[y, x])
            e = np.float64(data[y, x + 1]) + np.float64(data[y, x - 1])
            e = e / np.float64(2) - np.float64(data[y, x])
            out[y, x] = np.float32(-np.float64(2) * (d / csy2 + e / csx2) * np.float64(100))
    return out


def _run_numpy(data: np.ndarray,
               cellsize_x: Union[int, float],
               cellsize_y: Union[int, float],
               boundary: str = 'nan') -> np.ndarray:
    data = data.astype(np.float64)
    if boundary == 'nan':
        return _cpu(data, cellsize_x, cellsize_y)
    padded = _pad_array(data, 1, boundary)
    result = _cpu(padded, cellsize_x, cellsize_y)
    return result[1:-1, 1:-1]


def _run_dask_numpy(data: da.Array,
                    cellsize_x: Union[int, float],
                    cellsize_y: Union[int, float],
                    boundary: str = 'nan') -> da.Array:
    data = data.astype(np.float64)
    _func = partial(_cpu, cellsize_x=cellsize_x, cellsize_y=cellsize_y)
    out = data.map_overlap(_func,
                           depth=(1, 1),
                           boundary=_boundary_to_dask(boundary),
                           meta=np.array(()))
    return out


@cuda.jit(device=True)
def _gpu(arr, cellsize_x, cellsize_y):
    d = (arr[1, 0] + arr[1, 2]) / 2 - arr[1, 1]
    e = (arr[0, 1] + arr[2, 1]) / 2 - arr[1, 1]
    csx2 = cellsize_x * cellsize_x
    csy2 = cellsize_y * cellsize_y
    curv = -2 * (d / csy2 + e / csx2) * 100
    return curv


@cuda.jit
def _run_gpu(arr, cellsize_x, cellsize_y, out):
    i, j = cuda.grid(2)
    di = 1
    dj = 1
    if (i - di >= 0 and i + di <= out.shape[0] - 1 and
            j - dj >= 0 and j + dj <= out.shape[1] - 1):
        out[i, j] = _gpu(arr[i - di:i + di + 1, j - dj:j + dj + 1], cellsize_x, cellsize_y)


def _run_cupy(data: cupy.ndarray,
              cellsize_x: Union[int, float],
              cellsize_y: Union[int, float],
              boundary: str = 'nan') -> cupy.ndarray:
    if boundary != 'nan':
        padded = _pad_array(data, 1, boundary)
        result = _run_cupy(padded, cellsize_x, cellsize_y)
        return result[1:-1, 1:-1]

    data = data.astype(cupy.float64)
    csx = cupy.array([float(cupy.asnumpy(cellsize_x).item())], dtype='f8')
    csy = cupy.array([float(cupy.asnumpy(cellsize_y).item())], dtype='f8')

    griddim, blockdim = cuda_args(data.shape)
    out = cupy.empty(data.shape, dtype='f4')
    out[:] = cupy.nan

    _run_gpu[griddim, blockdim](data, csx, csy, out)

    return out


def _run_dask_cupy(data: da.Array,
                   cellsize_x: Union[int, float],
                   cellsize_y: Union[int, float],
                   boundary: str = 'nan') -> da.Array:
    data = data.astype(cupy.float64)
    csx = cupy.array([float(cupy.asnumpy(cellsize_x).item())], dtype='f8')
    csy = cupy.array([float(cupy.asnumpy(cellsize_y).item())], dtype='f8')

    _func = partial(_run_cupy, cellsize_x=csx, cellsize_y=csy)

    out = data.map_overlap(_func,
                           depth=(1, 1),
                           boundary=_boundary_to_dask(boundary, is_cupy=True),
                           meta=cupy.array(()))
    return out


@supports_dataset
def curvature(agg: xr.DataArray,
              name: Optional[str] = 'curvature',
              boundary: str = 'nan') -> xr.DataArray:
    """
    Calculates, for all cells in the array, the curvature (second
    derivative) of each cell based on the elevation of its neighbors
    in a 3x3 grid. A positive curvature indicates the surface is
    upwardly convex. A negative value indicates it is upwardly
    concave. A value of 0 indicates a flat surface.

    Units of the curvature output raster are one hundredth (1/100)
    of a z-unit.

    Parameters
    ----------
    agg : xarray.DataArray or xr.Dataset
        2D NumPy, CuPy, NumPy-backed Dask xarray DataArray of elevation values.
        Must contain `res` attribute.
        If a Dataset is passed, the operation is applied to each
        data variable independently.
    name : str, default='curvature'
        Name of output DataArray.
    boundary : str, default='nan'
        How to handle edges where the kernel extends beyond the raster.
        ``'nan'``     — fill missing neighbours with NaN (default).
        ``'nearest'`` — repeat edge values.
        ``'reflect'`` — mirror at boundary.
        ``'wrap'``    — periodic / toroidal.

    Returns
    -------
    curvature_agg : xarray.DataArray or xr.Dataset
        If `agg` is a DataArray, returns a DataArray of the same type.
        If `agg` is a Dataset, returns a Dataset with curvature computed
        for each data variable.
        2D aggregate array of curvature values.
        All other input attributes are preserved.

    References
    ----------
        - arcgis: https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/how-curvature-works.htm # noqa

    Examples
    --------
    Curvature works with NumPy backed xarray DataArray
    .. sourcecode:: python

        >>> import numpy as np
        >>> import dask.array as da
        >>> import xarray as xr
        >>> from xrspatial import curvature
        >>> flat_data = np.zeros((5, 5), dtype=np.float32)
        >>> flat_raster = xr.DataArray(flat_data, attrs={'res': (1, 1)})
        >>> flat_curv = curvature(flat_raster)
        >>> print(flat_curv)
        <xarray.DataArray 'curvature' (dim_0: 5, dim_1: 5)>
        array([[nan, nan, nan, nan, nan],
               [nan, -0., -0., -0., nan],
               [nan, -0., -0., -0., nan],
               [nan, -0., -0., -0., nan],
               [nan, nan, nan, nan, nan]])
        Dimensions without coordinates: dim_0, dim_1
        Attributes:
            res:      (1, 1)

    Curvature works with Dask with NumPy backed xarray DataArray
    .. sourcecode:: python

        >>> convex_data = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]], dtype=np.float32)
        >>> convex_raster = xr.DataArray(
            da.from_array(convex_data, chunks=(3, 3)),
            attrs={'res': (10, 10)}, name='convex_dask_numpy_raster')
        >>> print(convex_raster)
        <xarray.DataArray 'convex_dask_numpy_raster' (dim_0: 5, dim_1: 5)>
        dask.array<array, shape=(5, 5), dtype=float32, chunksize=(3, 3), chunktype=numpy.ndarray>
        Dimensions without coordinates: dim_0, dim_1
        Attributes:
            res:      (10, 10)
        >>> convex_curv = curvature(convex_raster, name='convex_curvature')
        >>> print(convex_curv)  # return a xarray DataArray with Dask-backed array
        <xarray.DataArray 'convex_curvature' (dim_0: 5, dim_1: 5)>
        dask.array<_trim, shape=(5, 5), dtype=float32, chunksize=(3, 3), chunktype=numpy.ndarray>
        Dimensions without coordinates: dim_0, dim_1
        Attributes:
            res:      (10, 10)
        >>> print(convex_curv.compute())
        <xarray.DataArray 'convex_curvature' (dim_0: 5, dim_1: 5)>
        array([[nan, nan, nan, nan, nan],
               [nan, -0.,  1., -0., nan],
               [nan,  1., -4.,  1., nan],
               [nan, -0.,  1., -0., nan],
               [nan, nan, nan, nan, nan]])
        Dimensions without coordinates: dim_0, dim_1
        Attributes:
            res:      (10, 10)

    Curvature works with CuPy backed xarray DataArray.
    .. sourcecode:: python

        >>> import cupy
        >>> concave_data = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]], dtype=np.float32)
        >>> concave_raster = xr.DataArray(
            cupy.asarray(concave_data),
            attrs={'res': (10, 10)}, name='concave_cupy_raster')
        >>> concave_curv = curvature(concave_raster)
        >>> print(type(concave_curv.data))
        <class 'cupy.core.core.ndarray'>
        >>> print(concave_curv)
        <xarray.DataArray 'curvature' (dim_0: 5, dim_1: 5)>
        array([[nan, nan, nan, nan, nan],
               [nan, -0., -1., -0., nan],
               [nan, -1.,  4., -1., nan],
               [nan, -0., -1., -0., nan],
               [nan, nan, nan, nan, nan]], dtype=float32)
        Dimensions without coordinates: dim_0, dim_1
        Attributes:
            res:      (10, 10)
    """
    _validate_raster(agg, func_name='curvature', name='agg')

    cellsize_x, cellsize_y = get_dataarray_resolution(agg)
    if (not np.isfinite(cellsize_x) or not np.isfinite(cellsize_y)
            or cellsize_x == 0 or cellsize_y == 0):
        raise ValueError(
            "curvature() requires a non-zero, finite cell size on both axes; "
            f"got cellsize_x={cellsize_x!r}, cellsize_y={cellsize_y!r}. "
            "Set agg.attrs['res'] to a (x, y) tuple of non-zero floats, "
            "or attach numeric x/y coordinates to the DataArray."
        )

    _validate_boundary(boundary)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=_run_numpy,
        cupy_func=_run_cupy,
        dask_func=_run_dask_numpy,
        dask_cupy_func=_run_dask_cupy
    )
    out = mapper(agg)(agg.data, cellsize_x, cellsize_y, boundary)
    return xr.DataArray(out,
                        name=name,
                        coords=agg.coords,
                        dims=agg.dims,
                        attrs=agg.attrs)
