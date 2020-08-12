"""
everything about geospatial corrdinates / locations / geometries ...
"""
import os
import time

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
from netCDF4._netCDF4 import Dataset
from pyproj import transform, CRS, Proj
from shapely.geometry import Polygon, Point

from utils import hydro_util


def spatial_join(points_file, polygons_file):
    """join polygons layer to point layer, add polygon which the point is in to the point """

    points = gpd.read_file(points_file)
    polys = gpd.read_file(polygons_file)
    # Check the data
    if not (points.crs == polys.crs):
        points = points.to_crs(polys.crs)

    # Make a spatial join
    join = gpd.sjoin(points, polys, how="inner", op="within")
    return join


def crd2grid(y, x):
    ux, indX0, indX = np.unique(x, return_index=True, return_inverse=True)
    uy, indY0, indY = np.unique(y, return_index=True, return_inverse=True)

    minDx = np.min(ux[1:] - ux[0:-1])
    minDy = np.min(uy[1:] - uy[0:-1])
    maxDx = np.max(ux[1:] - ux[0:-1])
    maxDy = np.max(uy[1:] - uy[0:-1])
    if maxDx > minDx * 2:
        print("skipped rows")
    #     indMissX=np.where((ux[1:]-ux[0:-1])>minDx*2)[0]
    #     insertX=(ux[indMissX+1]+ux[indMissX])/2
    #     ux=np.insert(ux,indMissX,insertX)
    if maxDy > minDy * 2:
        print("skipped coloums")
    #     indMissY=np.where((uy[1:]-uy[0:-1])>minDy*2)
    #     raise Exception('skipped coloums or rows')

    uy = uy[::-1]
    ny = len(uy)
    indY = ny - 1 - indY
    return (uy, ux, indY, indX)


def array2grid(data, *, lat, lon):
    (uy, ux, indY, indX) = crd2grid(lat, lon)
    ny = len(uy)
    nx = len(ux)
    if data.ndim == 2:
        nt = data.shape[1]
        grid = np.full([ny, nx, nt], np.nan)
        grid[indY, indX, :] = data
    elif data.ndim == 1:
        grid = np.full([ny, nx], np.nan)
        grid[indY, indX] = data
    return grid, uy, ux


def trans_points(from_crs, to_crs, pxs, pys):
    """put the data into dataframe so that the speed of processing could be improved obviously
    :param
    pxs: x of every point (list/array)
    pys: y of every point (list/array)
    :return
    pxys_out: x and y compared a pair list to initialize a polygon
    """
    df = pd.DataFrame({'x': pxs, 'y': pys})
    start = time.time()
    df['x2'], df['y2'] = transform(from_crs, to_crs, df['x'].tolist(), df['y'].tolist())
    end = time.time()
    print('time consuming：', '%.7f' % (end - start))
    # after transforming xs and ys, pick out x2, y2，and tranform to numpy array，then do a transportation. Finally put coordination of every row to a list
    arr_x = df['x2'].values
    arr_y = df['y2'].values
    pxys_out = np.stack((arr_x, arr_y), 0).T
    return pxys_out


def trans_polygon(from_crs, to_crs, polygon_from):
    """transform coordination of every point of a polygon to one in a given coordination system"""
    polygon_to = Polygon()
    # 多边形外边界的各点坐标list里面是tuple
    boundary = polygon_from.boundary
    boundary_type = boundary.geom_type
    print(boundary_type)
    if boundary_type == 'LineString':
        pxs = polygon_from.exterior.xy[0]
        pys = polygon_from.exterior.xy[1]
        pxys_out = trans_points(from_crs, to_crs, pxs, pys)
        polygon_to = Polygon(pxys_out)
    elif boundary_type == 'MultiLineString':
        # if there is interior boundary in a polygon，then we need to transform its coordinations. Notice: maybe multiple interior boundaries exist.
        exts_x = boundary[0].xy[0]
        exts_y = boundary[0].xy[1]
        pxys_ext = trans_points(from_crs, to_crs, exts_x, exts_y)

        pxys_ints = []
        for i in range(1, len(boundary)):
            ints_x = boundary[i].xy[0]
            ints_y = boundary[i].xy[1]
            pxys_int = trans_points(from_crs, to_crs, ints_x, ints_y)
            pxys_ints.append(pxys_int)

        polygon_to = Polygon(shell=pxys_ext, holes=pxys_ints)
    else:
        print("error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    return polygon_to


def write_shpfile(geodata, output_folder, id_str="hru_id"):
    """generate a shpfile from geodataframe，the name is id of the pandas dataframe"""
    # Create a output path for the data
    gage_id = geodata.iloc[0, :][id_str]
    # id is number，here turn it to str
    output_file = str(int(gage_id)).zfill(8)
    output_fp = os.path.join(output_folder, output_file + '.shp')
    # Write those rows into a new file (the default output file format is Shapefile)
    geodata.to_file(output_fp)


def trans_shp_coord(input_folder, input_shp_file, output_folder,
                    output_crs_epsg_or_proj4_str='4326'):
    """tranform a shapefile to a target coord，default target coord is WGS84:  +proj=longlat +datum=WGS84 +no_defs"""
    # Join folder path and filename
    fp = os.path.join(input_folder, input_shp_file)
    data = gpd.read_file(fp)
    # crs_proj4 = CRS(data.crs).to_proj4()
    crs_proj4 = CRS(data.crs)
    # crs_final = CRS.from_proj4(output_crs_proj4_str)
    # Proj must be used，if not, maybe x represent longitude, and the other represent latitude and it's wrong
    crs_final = Proj(init='epsg:' + output_crs_epsg_or_proj4_str)
    # crs_final = CRS.from_epsg(output_crs_epsg_or_proj4_str)
    all_columns = data.columns.values  # ndarray type
    new_datas = []
    start = time.time()
    for i in range(0, data.shape[0]):  # data.shape[0]
        print("the  ", i, "st basin's shapefile:")
        newdata = gpd.GeoDataFrame()
        for column in all_columns:
            # when read shapefile using geodataframe, the name of geo column is "geometry"
            if column == 'geometry':
                # first change the coord
                polygon_from = data.iloc[i, :]['geometry']
                polygon_to = trans_polygon(crs_proj4, crs_final, polygon_from)
                # assign value to location i of newdata，if not it will be geoseries，which cannot be imported to shapefile
                newdata.at[0, 'geometry'] = polygon_to
                print(type(newdata.at[0, 'geometry']))
            else:
                newdata.at[0, column] = data.iloc[i, :][column]
        print("coordination transform！")
        print(newdata)
        # must use fiona's crs to guarantee the result is correct
        newdata.crs = fiona.crs.from_epsg(int(output_crs_epsg_or_proj4_str))
        print("Coordination system: ", newdata.crs)
        new_datas.append(newdata)
        write_shpfile(newdata, output_folder)
    end = time.time()
    print('time consuming：', '%.7f' % (end - start))
    return new_datas


def nearest_point_index(crs_from, crs_to, lon, lat, xs, ys):
    # x and y are proj coord，lon, lat should be transformed (x is longtitude projection，y is lat)
    x, y = transform(crs_from, crs_to, lon, lat)
    index_x = (np.abs(xs - x)).argmin()
    index_y = (np.abs(ys - y)).argmin()
    return [index_x, index_y]


def create_mask(poly, xs, ys, lons, lats, crs_from, crs_to):
    """根据只有一个Polygon的shapefile和一个netcdf文件的所有坐标，生成该shapefile对应的mask。用xy坐标或经纬度都可以，先用经纬度测试下
       因为netcdf代表的空间太大，所以为了计算较快，直接生成索引比较合适，即取netcdf的变量中的合适点的index"""
    mask_index = []
    # 首先想办法减少循环的范围，然后在循环内使用map实现快速循环，现在思路是这样的：
    # 每行选出一个最接近的index组成一个INDEX集合，首先读取bound的范围，转换到x和y上判断范围
    poly_bound = poly.bounds
    poly_bound_min_lat = poly_bound[1]
    poly_bound_min_lon = poly_bound[0]
    poly_bound_max_lat = poly_bound[3]
    poly_bound_max_lon = poly_bound[2]
    index_min = nearest_point_index(crs_from, crs_to, poly_bound_min_lon, poly_bound_min_lat, xs, ys)
    index_max = nearest_point_index(crs_from, crs_to, poly_bound_max_lon, poly_bound_max_lat, xs, ys)
    # 注意y是倒序的
    range_x = [index_min[0], index_max[0]]
    range_y = [index_max[1], index_min[1]]
    # 注意在nc文件中，lat和lon的坐标都是(y,x)，所以顺着lons和lats的坐标轴循环，第一个先循环y到range_y[1] + 1（range_x,range_y都是闭区间）
    for i in range(range_y[0], range_y[1] + 1):
        for j in range(range_x[0], range_x[1] + 1):
            if is_point_in_boundary(lons[i][j], lats[i][j], poly):
                mask_index.append((i, j))
    return mask_index


def is_point_in_boundary(px, py, poly):
    """给定一个点的经纬度坐标，判断是否在多多边形边界内
    :param
    polygon--shapely.geometry.Polygon
    """
    point = Point(px, py)
    return point.within(poly)


def calc_avg(mask, netcdf_data, var_type):
    """有了mask之后，就可以直接取对应位置的数据了，mask是一个包含二维tuple的list"""

    # var_data是一个三维的数组，var_data的时间变量是第1个维度，利用mask的数据搜索另外两维，一次性读取数据量太大，无法读入，因此要对时间循环，循环后直接根据索引取数据
    # 数据的索引花的时间太久，感觉必须要在下载数据时就先按照范围把数据下载好，然后再来生成mask，或者用matlab看看读取速度会不会较快
    mask = np.array(mask)
    index = mask.T

    def f_avg(i):
        data_day = netcdf_data.variables[var_type][i]
        data_chosen = data_day[index[0], index[1]]
        # 直接numpy指定位置处的数据求平均
        data_mean = np.mean(data_chosen, axis=0)
        return data_mean

    # 使用map循环
    all_mean_data = list(map(f_avg, range(365)))

    return all_mean_data


def shps_trans_coord(input_folder, output_folder):
    """transform coords of all shapefiles in the folder--"input_folder",
       and save the results in the folder--"output_folder"
    """
    # Define path to folder，以r开头表示相对路径
    shp_file_names = []
    for f_name in os.listdir(input_folder):
        if f_name.endswith('.shp'):
            shp_file_names.append(f_name)

    for i in range(len(shp_file_names)):
        shp_file = shp_file_names[i]
        # output_folder = r"examples_data/wgs84lccsp2" crs_final_str = '+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5
        # +lon_0=-100 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
        new_datas = trans_shp_coord(input_folder, shp_file, output_folder)


def basin_avg_netcdf(netcdf_file, shp_file, mask_file):
    data_netcdf = Dataset(netcdf_file, 'r')  # reads the netCDF file
    # 看看netcdf的格式具体是什么样的，便于后面判断坐标之间的空间关系
    print(data_netcdf)
    # get all variable names
    print(data_netcdf.variables.keys())
    temp_lat = data_netcdf.variables['lat']  # temperature variable
    temp_lon = data_netcdf.variables['lon']  # temperature variable
    for d in data_netcdf.dimensions.items():
        print(d)
    x, y = data_netcdf.variables['x'], data_netcdf.variables['y']
    # x，y是其他变量的坐标：lat(y,x), lon(y,x), prcp(time,y,x)。所以先看看x和y的数据的规律
    x = data_netcdf.variables['x'][:]
    y = data_netcdf.variables['y'][:]
    # 判断x和y是否递增，根据下面的all函数判断可知：x是递增的，y是递减的，因为只有递增递减的，才能先根据边界把区域定好，然后再分析mask，这样会很快
    lx = list(x)
    ly = list(y)
    print(all(ix < jx for ix, jx in zip(lx, lx[1:])))
    print(all(iy > jy for iy, jy in zip(ly, ly[1:])))
    lons = data_netcdf.variables['lon'][:]
    lats = data_netcdf.variables['lat'][:]

    # 投影坐标系（从官网得到的daymet坐标信息）
    crs_pro_str = '+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
    # 地理坐标系（epsg 4326 对应的str）
    crs_geo_str = '+proj=longlat +datum=WGS84 +no_defs'
    # 后面create_mask时候需要将经纬度坐标转换为投影坐标，所以这里是crs_geo_str作crs_from
    crs_from = CRS.from_proj4(crs_geo_str)
    crs_to = CRS.from_proj4(crs_pro_str)

    # 先选择一个shpfile
    new_shps = gpd.read_file(shp_file)
    polygon = new_shps.at[0, 'geometry']
    start = time.time()
    mask = create_mask(polygon, x, y, lons, lats, crs_from, crs_to)
    end = time.time()
    print('生成mask耗时：', '%.7f' % (end - start))
    hydro_util.serialize_numpy(np.array(mask), mask_file)
    var_types = ['tmax']
    # var_types = ['tmax', 'tmin', 'prcp', 'srad', 'vp', 'swe', 'dayl']
    avgs = []
    for var_type in var_types:
        start = time.time()
        avg = calc_avg(mask, data_netcdf, var_type)
        end = time.time()
        print('计算耗时：', '%.7f' % (end - start))
        print('平均值：', avg)
        avgs.append(avg)

    return avgs
