##### NOTE: NEED TO UPDATE PATHNAMES FOR YOUR SYSTEM
naip_dir = "C:\\Users\\ME\\Dropbox\\HCSproject\\data\\PoC\\app_kalbar_cntk\\"
import os
import subprocess
import gdal
filename_base = "app_kalbar_input_ndvi_s2"
nlcd_filepath = naip_dir + 'app_kalbar_remap.tif'
output_dir = naip_dir + "tiles\\"


img = gdal.Open(os.path.join(naip_dir, '{}.tif'.format(filename_base)))
out = os.path.join(naip_dir, '{}_tiled.tif'.format(filename_base))
translateoptions = gdal.TranslateOptions(gdal.ParseCommandLine("-co COMPRESS=LZW, TILED=YES"))
gdal.Translate(out, img, options = translateoptions)
naip_filepath = out

from osgeo import gdal
from gdalconst import *
os.environ['PROJ_LIB']='C:\\Users\\ME\\Anaconda3\\envs\\spatial_ml\\Library\\share\\'
import osr
from mpl_toolkits.basemap import Basemap
from collections import namedtuple

LatLonBounds = namedtuple('LatLonBounds', ['llcrnrlat', 'llcrnrlon', 'urcrnrlat', 'urcrnrlon'])


def get_bounding_box(naip_filepath):
    ''' Finds a bounding box for the NAIP GeoTIFF in lat/lon '''
    naip_image = gdal.Open(naip_filepath, GA_ReadOnly)
    naip_proj = osr.SpatialReference()
    naip_proj.ImportFromWkt(naip_image.GetProjection())
    naip_ulcrnrx, naip_xstep, _, naip_ulcrnry, _, naip_ystep = naip_image.GetGeoTransform()

    world_map = Basemap(lat_0=0,
                        lon_0=0,
                        llcrnrlat=-90, urcrnrlat=90,
                        llcrnrlon=-180, urcrnrlon=180,
                        resolution='c', projection='stere')
    world_proj = osr.SpatialReference()
    world_proj.ImportFromProj4(world_map.proj4string)
    ct_to_world = osr.CoordinateTransformation(naip_proj, world_proj)

    lats = []
    lons = []
    for corner_x, corner_y in [(naip_ulcrnrx, naip_ulcrnry),
                               (naip_ulcrnrx, naip_ulcrnry + naip_image.RasterYSize * naip_ystep),
                               (naip_ulcrnrx + naip_image.RasterXSize * naip_xstep,
                                naip_ulcrnry + naip_image.RasterYSize * naip_ystep),
                               (naip_ulcrnrx + naip_image.RasterXSize * naip_xstep, naip_ulcrnry)]:
        xpos, ypos, _ = ct_to_world.TransformPoint(corner_x, corner_y)
        lon, lat = world_map(xpos, ypos, inverse=True)
        lats.append(lat)
        lons.append(lon)

    return (LatLonBounds(llcrnrlat=min(lats),
                         llcrnrlon=min(lons),
                         urcrnrlat=max(lats),
                         urcrnrlon=max(lons)))


region_bounds = get_bounding_box(naip_filepath)

RegionSize = namedtuple('RegionSize', ['width', 'height'])  # in meters!
import numpy as np

def get_approx_region_size(region_bounds):
    ''' Returns the region width (at mid-lat) and height in meters'''
    mid_lat_radians = (region_bounds.llcrnrlat + region_bounds.urcrnrlat) * \
                      (np.pi / 360)
    earth_circumference = 6.371E6 * 2 * np.pi # in meters
    region_middle_width_meters = (region_bounds.urcrnrlon - region_bounds.llcrnrlon) * \
                                 earth_circumference * np.cos(mid_lat_radians) / 360
    region_height_meters = (region_bounds.urcrnrlat - region_bounds.llcrnrlat) * \
                           earth_circumference / 360
    return(RegionSize(region_middle_width_meters, region_height_meters))

approx_region_size = get_approx_region_size(region_bounds)

from PIL import Image


def create_helper_functions(region_bounds, nlcd_filepath, naip_filepath):
    ''' Makes helper functions to label points (NLCD) and extract tiles (NAIP) '''
    nlcd_image = gdal.Open(nlcd_filepath, GA_ReadOnly)
    nlcd_proj = osr.SpatialReference()
    nlcd_proj.ImportFromWkt(nlcd_image.GetProjection())
    nlcd_ulcrnrx, nlcd_xstep, _, nlcd_ulcrnry, _, nlcd_ystep = nlcd_image.GetGeoTransform()

    naip_image = gdal.Open(naip_filepath, GA_ReadOnly)
    naip_proj = osr.SpatialReference()
    naip_proj.ImportFromWkt(naip_image.GetProjection())
    naip_ulcrnrx, naip_xstep, _, naip_ulcrnry, _, naip_ystep = naip_image.GetGeoTransform()

    region_map = Basemap(lat_0=(region_bounds.llcrnrlat + region_bounds.urcrnrlat) / 2,
                         lon_0=(region_bounds.llcrnrlon + region_bounds.urcrnrlon) / 2,
                         llcrnrlat=region_bounds.llcrnrlat,
                         llcrnrlon=region_bounds.llcrnrlon,
                         urcrnrlat=region_bounds.urcrnrlat,
                         urcrnrlon=region_bounds.urcrnrlon,
                         resolution='c',
                         projection='stere')

    region_proj = osr.SpatialReference()
    region_proj.ImportFromProj4(region_map.proj4string)
    ct_to_nlcd = osr.CoordinateTransformation(region_proj, nlcd_proj)
    ct_to_naip = osr.CoordinateTransformation(region_proj, naip_proj)

    def get_nlcd_label(point):
        ''' Project lat/lon point to NLCD GeoTIFF; return label of that point '''
        basemap_coords = region_map(point.lon, point.lat)  # NB unusual argument order#
        # x, y, _ = [int(i) for i in ct_to_nlcd.TransformPoint(*basemap_coords)]
        ####### NOTE: ORIGINAL CODE USED LINE ABOVE. ROUNDING SEEMED TO CAUSE PROBLEMS, BUT NOW LATER CODE WONT WORK. MAYBE ITS AN ISSUE
        ####### WITH THE SHIFT IN RESOLUTIONS FROM THE NAIP/NLDC APPLICATION TO OUR APPLICATION

        x, y, _ = [i for i in ct_to_nlcd.TransformPoint(*basemap_coords)]
        xoff = int(round((x - nlcd_ulcrnrx) / nlcd_xstep))
        yoff = int(round((y - nlcd_ulcrnry) / nlcd_ystep))
        label = int(nlcd_image.ReadAsArray(xoff=xoff, yoff=yoff, xsize=1, ysize=1))
        return (label)

    def get_naip_tile(tile_bounds, tile_size):
        ''' Check that tile lies within county bounds; if so, extract its image '''

        # Transform tile bounds in lat/lon to NAIP projection coordinates
        xmax, ymax = region_map(tile_bounds.urcrnrlon, tile_bounds.urcrnrlat)
        xmin, ymin = region_map(tile_bounds.llcrnrlon, tile_bounds.llcrnrlat)
        xstep = (xmax - xmin) / tile_size.width
        ystep = (ymax - ymin) / tile_size.height

        grid = np.mgrid[xmin:xmax:tile_size.width * 1j, ymin:ymax:tile_size.height * 1j]
        shape = grid[0, :, :].shape
        size = grid[0, :, :].size
        xy_target = np.array(ct_to_naip.TransformPoints(grid.reshape(2, size).T))
        xx = xy_target[:, 0].reshape(shape)
        yy = xy_target[:, 1].reshape(shape)

        # Extract rectangle from NAIP GeoTIFF containing superset of needed points
        xoff = int(round((xx.min() - naip_ulcrnrx) / naip_xstep))
        yoff = int(round((yy.max() - naip_ulcrnry) / naip_ystep))
        xsize_to_use = int(np.ceil((xx.max() - xx.min()) / np.abs(naip_xstep))) + 1
        ysize_to_use = int(np.ceil((yy.max() - yy.min()) / np.abs(naip_ystep))) + 1
        data = naip_image.ReadAsArray(xoff=xoff,
                                      yoff=yoff,
                                      xsize=xsize_to_use,
                                      ysize=ysize_to_use)
        # Map the pixels of interest in NAIP GeoTIFF to the tile (might involve rotation or scaling)
        image = np.zeros((xx.shape[1], xx.shape[0], 3)).astype(float)  # rows are height, cols are width, third dim is color
        #image = np.zeros((xx.shape[1], xx.shape[0], 1)).astype(int)  # rows are height, cols are width, third dim is color

        try:
            for i in range(xx.shape[0]):
                for j in range(xx.shape[1]):
                    x_idx = int(round((xx[i, j] - naip_ulcrnrx) / naip_xstep)) - xoff
                    y_idx = int(round((yy[i, j] - naip_ulcrnry) / naip_ystep)) - yoff
                    # image[xx.shape[1] - j - 1, i, :] = data[:, y_idx, x_idx]
                    image[xx.shape[1] - j - 1, i, :] = data[y_idx, x_idx]
        except TypeError as e:
            # The following can occur if our pixel superset request exceeds the GeoTIFF's bounds
            print("Out of Bounds")
            return (None)

        # if np.sum(image.sum(axis=2) == 0) > 10: # too many nodata pixels
        if np.sum(image > 0) == 0:  # too many nodata pixels
            print("All NoData")
            return None

        if np.sum(image == 0) > 20000:  # too many nodata pixels
            print("Too Many NoData")
            return None

        # image = Image.fromarray(image.astype('uint8'))
        image = (image * 255).astype(np.uint8)
        image[:, :, 1] = image[:, :, 0]
        image[:, :, 2] = image[:, :, 0]
        image = Image.fromarray((image * 255).astype(np.uint8))
        return (image)

    return (get_nlcd_label, get_naip_tile)


get_nlcd_label, get_naip_tile = create_helper_functions(region_bounds, nlcd_filepath, naip_filepath)

LatLonPosition = namedtuple('LatLonPosition', ['lat', 'lon'])
Tile = namedtuple('Tile', ['bounds', 'label'])

nlcd_label_to_class = {0: 'NA',
                       1: 'Not_HCSA',
                       2: 'HCSA'}


def find_tiles_with_consistent_labels(region_bounds, region_size, tile_size):
    ''' Find tiles for which nine grid points all have the same label '''
    tiles_wide = int(np.floor(region_size.width / tile_size.width))
    tiles_tall = int(np.floor(region_size.height / tile_size.height))
    tile_width = (region_bounds.urcrnrlon - region_bounds.llcrnrlon) / (region_size.width / tile_size.width)
    tile_height = (region_bounds.urcrnrlat - region_bounds.llcrnrlat) / (region_size.height / tile_size.height)

    current_lat = region_bounds.llcrnrlat
    current_lon = region_bounds.llcrnrlon

    tiles_to_use = []
    for i in range(tiles_tall):
        for j in range(tiles_wide):
            try:
                labels = []
                for k in range(3):
                    for ell in range(3):
                        my_label = get_nlcd_label(LatLonPosition(lat=current_lat + tile_width * (1 + 2 * k) / 6,
                                                                 lon=current_lon + tile_height * (1 + 2 * ell) / 6))
                        labels.append(nlcd_label_to_class[my_label])
                num_matching = np.sum(np.array(labels) == labels[4])
                if (num_matching == 9):
                    bounds = LatLonBounds(llcrnrlat=current_lat,
                                          llcrnrlon=current_lon,
                                          urcrnrlat=current_lat + tile_height,
                                          urcrnrlon=current_lon + tile_width)
                    tiles_to_use.append(Tile(bounds=bounds,
                                             label=labels[4]))
            except KeyError:
                pass
            current_lon += tile_width
        current_lon = region_bounds.llcrnrlon
        current_lat += tile_height
    return (tiles_to_use)


tile_size = RegionSize(120, 120)
tiles = find_tiles_with_consistent_labels(region_bounds, approx_region_size, tile_size)
print('Found {} tiles to extract'.format(len(tiles)))


def extract_tiles(tiles, dest_folder, filename_base):
    ''' Coordinates saving tile data, including extracted images and CSV descriptions '''
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    tile_descriptions = []
    i = 0
    while i < len(tiles):
        tile = tiles[i]
        print(tile)
        tile_image = get_naip_tile(tile.bounds, tile_size)
        i += 1
        if (tile_image is None):
            continue  # tile did not lie entirely within the county boundary (it was at least partially blank)
        my_directory = os.path.join(dest_folder, '{}'.format(tile.label))
        my_filename = os.path.join(my_directory, '{}_{}.png'.format(filename_base, i))
        print(my_filename)
        if not os.path.exists(my_directory):
            os.makedirs(my_directory)
        tile_image.save(my_filename, 'PNG')
    return


extract_tiles(tiles, output_dir, filename_base)