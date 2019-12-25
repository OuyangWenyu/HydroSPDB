"""
everything about geospatial corrdinates / locations / geometries ...
"""
import geopandas as gpd


def spatial_join(points_file, polygons_file):
    """join polygons layer to point layer, add polygon which the point is in to the point """

    points = gpd.read_file(points_file)
    print("Number of rows:", len(points))

    polys = gpd.read_file(polygons_file)
    # Check the data
    print("Number of rows:", len(polys))

    if not (points.crs == polys.crs):
        points = points.to_crs(polys.crs)

    # Make a spatial join
    join = gpd.sjoin(points, polys, how="inner", op="within")
    return join
