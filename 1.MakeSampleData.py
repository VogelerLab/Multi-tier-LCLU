########################################################################################################################
# Title: LCLUC project data extraction script
# --------------------------------------------
# Description: This script will extract and process a set of yearly features for a set of input points from various
# remote sensing datasets including Sentinel-1, Sentinel-2, SRTM, etc.
# Sentinel-2 data is assumed to be the TOA dataset and is processed for cloud and cloud-shadow masking using a
# supplementary script. Sentinel-1 data is also processed and speckle-filtered using another script. For each year,
# the final processed data is aggregated and packed in a list of numpy arrays and saved in python pickle format.
# The output file will be used by feature analysis and selection script (AnalyzeFeatures.py)
#
# Required input data: Sampling points dataset stored under Google Earth Engine (GEE) repository
# Generated output file(s): Extracted remote sensing data for sampling points stored in a pickle binary file (.p)
# plus a companion .txt file for parameters specification
#
# Note: Variables in the variables setting block are set to the values used in the last program run and may not
# represent the values used to create the files given in the SampleData folder.
########################################################################################################################
# Shahriar S. Heydari, Vogeler Lab, 4/24/2024

import ee
import pickle, time
import numpy as np
import Modules_Sen2Utilities
import Modules_GEE_S1_ARD

ee.Initialize()

# Variables setting block
##########################
script_version = '6.4'
out_file_path = ''
out_file_name = 'SA_urbanValidationSet'
# The points of interest should already be placed in a GEE feature collection declared here. It is assumed
# that they have a field named 'ID' as point identifier.
GEEdatasetName = 'users/sheydari/Africa_samples/SA_urbanValidationSet'
points = ee.FeatureCollection(GEEdatasetName).sort('ID')
pointsList = points.toList(10000).getInfo()
# Sentinel-2 specific processing parameters:
cloudy_scene_percentage_threshold = 30
minimum_S2_scenes = 5
Modules_Sen2Utilities.CLD_PRB_THRESH = 30
Modules_Sen2Utilities.NIR_DRK_THRESH = .15
Modules_Sen2Utilities.CLD_PRJ_DIST = 15
Modules_Sen2Utilities.BUFFER = 200
# Data extraction years:
start_year = 2016         # start year for feature generation, inclusive
end_year = 2021           # end year, exclusive
# Extracted bands declaration:
Sentinel2_bands = ['B2','B3','B4','B8','B11','B12']
Sentinel1_bands = ['VV']
# Bands to be included in collecting zonal statistics (min, max, stdev) over 3x3 and 5x5 windows:
spatial_zonalStat_bands = ['VV','B3','B8','tcg', 'tcw', 'tcb']
monthNames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
# GLCM texture features to collect from Sentinel-1 and Sentinel-2 data:
S1_GLCM_band = 'VV'
S1_GLCM_q_level = 64
S1_GLCM_input_min = -30
S1_GLCM_input_max = 30
S1_GLCM_features = [
    {'radius': 5, 'metrics': ['_asm','_contrast','_corr','_savg','_imcorr1','_imcorr2','_shade','_prom']},
    {'radius': 9, 'metrics': ['_asm','_contrast','_corr','_savg','_imcorr1','_imcorr2','_shade','_prom']},
]
S2_GLCM_band = 'B8'     # NIR
S2_GLCM_q_level = 64
S2_GLCM_input_min = 0
S2_GLCM_input_max = 1
S2_GLCM_features = [
    {'radius': 5, 'metrics': ['_asm','_contrast','_corr','_savg','_imcorr1','_imcorr2','_shade','_prom']},
    {'radius': 9, 'metrics': ['_asm','_contrast','_corr','_savg','_imcorr1','_imcorr2','_shade','_prom']},
]
# Parameters to be used by Sentinel-1 enhancement module
S1_enhancement = True
S1_enhancement_parameters = {
    # 1. Data Selection
    'START_DATE': None,     # should be set before calling the procedure
    'STOP_DATE': None,      # should be set before calling the procedure
    'GEOMETRY': None,       # should be set before calling the procedure
    'POLARIZATION': 'VV',
    'ORBIT': 'BOTH',
    # 2. Additional Border noise correction
    'APPLY_ADDITIONAL_BORDER_NOISE_CORRECTION': True,
    # 3.Speckle filter
    'APPLY_SPECKLE_FILTERING': True,
    'SPECKLE_FILTER_FRAMEWORK': 'MULTI',
    'SPECKLE_FILTER': 'LEE',
    'SPECKLE_FILTER_KERNEL_SIZE': 15,
    'SPECKLE_FILTER_NR_OF_IMAGES': 10,
    # 4. Radiometric terrain normalization
    'APPLY_TERRAIN_FLATTENING': True,
    'DEM': ee.Image('USGS/SRTMGL1_003'),
    'TERRAIN_FLATTENING_MODEL': 'VOLUME',
    'TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER': 0,
    # 5. Output
    'FORMAT': 'DB',
    'CLIP_TO_ROI': False,
    'SAVE_ASSETS': False
}
# Additional datasets
wclim = ee.Image("users/sheydari/WorldClim_2_1").select([*range(19)])   # select 19 bio-climate variables
SRTM = ee.Image("USGS/SRTMGL1_003")                                     # use SRTM DEM to create topography variables
terrainBands = ee.Terrain.products(SRTM)
# RESOLVE is a global designator for ecoregions. Note that RESOLVE ecoregions cover terrestrial land, not water bodies.
RESOLVE_ecoregions = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017")
ecoBand = RESOLVE_ecoregions.reduceToImage(properties=['ECO_ID'], reducer=ee.Reducer.first())
# CHILI index ranging from 0 (very cool) to 255 (very warm).
CHILI_index = ee.Image("CSP/ERGo/1_0/Global/SRTM_CHILI")
# iSDAsoil is a soil texture dataset which labels each 30m-resolution pixel with one of 12 USDA soil types
iSDAsoil = ee.Image("ISDASOIL/Africa/v1/texture_class").select(0)
# TerraClimate is a coarse-resolution monthly dataset of climate variables
terraClim = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE')
### other parameters
scale = 10                  # scale used to export remote sensing data
min_ntl_obs = 2             # minimum required number of nigh-time light observations per month

##########################################################################################
# Helper functions
##########################################################################################

def addDate(image):
    # Add the image date as THE LAST band to it in fractional year format
    def msToFrac(ms):
        # Convert milliseconds since 1970-01-01 to fractional year
        year = (ee.Date(ms).get('year'))
        frac = (ee.Date(ms).getFraction('year'))
        return year.add(frac)

    proj = image.select(0).projection()
    image_date = image.date()
    dateBand = ee.Image.constant(msToFrac(image_date)).float().reproject(proj).rename('timestamp')
    image = image.addBands(dateBand)
    return image

def band_scale(image):
    # scale Sentinel-2 bands
    img = image.select(Sentinel2_bands).divide(10000)
    return image.addBands(img, overwrite=True).select(Sentinel2_bands)

def check_S2(image_collection, cloud_threshold):
    # Filter low quality and different-tile images
    collection = image_collection\
        .filterMetadata('GENERAL_QUALITY','equals','PASSED')\
        .filterMetadata('SENSOR_QUALITY','equals','PASSED')\
        .filterMetadata('GEOMETRIC_QUALITY','equals','PASSED')\
        .filterMetadata('RADIOMETRIC_QUALITY','equals','PASSED')\
        .filterMetadata('FORMAT_CORRECTNESS','equals','PASSED')\
        .filterMetadata('CLOUDY_PIXEL_PERCENTAGE','less_than',cloud_threshold)
    output_images = []
    collection = collection.toList(1000)
    for index in range(collection.length().getInfo()):#indices:
        picked_image = ee.Image(collection.get(index))
        # mask fringe pixels (in some cases, there are non-masked strips in some bands around the tile footprint
        imageArrayMask = picked_image.toArray().mask()
        picked_image = picked_image.mask(imageArrayMask)
        output_images.append(picked_image)
    return ee.ImageCollection(output_images)

def gen_GLCM(image, GLCM_band, GLCM_q_level, GLCM_input_min, GLCM_input_max, GLCM_features):
    # Generate GLCM features (as defined by GLCM_features variable) for the input image

    def genGLCM(q_image, feature):
        # function to generate requested set of GLCM metrics for a specific band
        GLCM_f = [GLCM_band + x for x in feature['metrics']]
        sel_GLCM_features = [GLCM_band + x + '_' + str(feature['radius']) + 'x' + str(GLCM_q_level) for x in feature['metrics']]
        return q_image.glcmTexture(size=feature['radius']).select(GLCM_f).rename(sel_GLCM_features)

    image = image.unitScale(GLCM_input_min, GLCM_input_max).multiply(GLCM_q_level).toInt()
    # start with the image itself, add GLCM bands to it, and drop the first band at last
    q_image = image
    for feature in GLCM_features:
        # calculate and add glcm metrics for a specific glcm feature
        q_image = q_image.addBands(genGLCM(image, feature))
    return q_image.slice(1)

def buildAndSampleStacks(image_collection, type, point):
    # This function gets an S1 or S2 image collection and returns a numpy array corresponding to the sampled pixel
    # time series

    # converting masked band values to zero. It is required because otherwise conversion to array will make all pixel
    # values (valid or invalid) to be masked.
    image_collection = image_collection.map(lambda image: image.unmask())
    # convert image collection to image array, for both center pixel (LandsatStack) and neighborhood data (spatialStack)
    imageStack = image_collection.toArray()

    # each set of collected values will be named with corresponding band names. These name lists will be recorded
    # in the output configuration file for easy interpretation of data when reading it later
    bandNames = image_collection.first().bandNames().getInfo()
    if type == 'S2':
        prinicipalBandsIndex = [ind for ind, u in enumerate(bandNames) if u in Sentinel2_bands]
    elif type == 'S1':
        prinicipalBandsIndex = [ind for ind, u in enumerate(bandNames) if u in Sentinel1_bands]
    else:
        prinicipalBandsIndex = [ind for ind, u in enumerate(bandNames)]

    try:
        pixels = imageStack.sample(region=point, scale=scale).getInfo()
        pixel_array_f = np.array(pixels['features'][0]['properties']['array'])
        # drop readings with zero entries on principal six bands
        nonzero_index = np.where(~np.all(pixel_array_f[:,prinicipalBandsIndex] == 0, axis=1))[0]
        pixel_array = pixel_array_f[nonzero_index, :]

    except:
        pixel_array = []

    return pixel_array, bandNames

##########################################################################################
# Main program
##########################################################################################

point_data = []     # python list to hold the generated data
failed_points = []  # python list to hold the failed points/years during GEE data extraction
start_time = time.time()

# main loop over sample points
for p in range(len(pointsList)):
    print('Processing row# {}'.format(p))
    point = ee.Feature(pointsList[p])
    geometry = point.geometry()
    [lon, lat] = geometry.coordinates().getInfo()
    point_ID = point.get('ID').getInfo()
    point = ee.FeatureCollection(point)

    # 1. Generate yearly-variable features
    #######################################

    variable_data = []
    print(' - generate yearly-variable data structures.')
    for year in range(start_year, end_year):

        # extract Sentinel-2 time series
        ################################
        image_collection = ee.ImageCollection("COPERNICUS/S2").filterBounds(geometry)\
            .filterDate(str(year)+'-01-01', str(year+1)+'-01-01')
        # process Sentinel-2 time series for scenes cloudiness and filter highly cloudy scenes
        S2_original_scene_list = image_collection.aggregate_array('system:index').getInfo()
        filtered_image_collection = check_S2(image_collection, cloudy_scene_percentage_threshold)
        S2_filtered_scene_list = filtered_image_collection.aggregate_array('system:index').getInfo()
        new_cloud_threshold = cloudy_scene_percentage_threshold
        while (len(S2_filtered_scene_list) < minimum_S2_scenes) and (new_cloud_threshold <= 50):
            new_cloud_threshold += 10
            filtered_image_collection = check_S2(image_collection, new_cloud_threshold)
            S2_filtered_scene_list = filtered_image_collection.aggregate_array('system:index').getInfo()
        if new_cloud_threshold >= 60:
            print(' *** More than 60% of each scene for year {} is reported as cloudy and therefroe the year skipped ***'.format(year))
            failed_points.append([point_ID, 'Sentinel-2 time series - year '+str(year)])
            continue

        if new_cloud_threshold != cloudy_scene_percentage_threshold:
            print(' - Cloudy scene threshold increased to {} for this point at year {}'.format(new_cloud_threshold, year))
        Sentinel2_image_collection = filtered_image_collection.sort('system:time_start').map(band_scale)\
            .map(Modules_Sen2Utilities.Sen2CloudMasking).map(Modules_Sen2Utilities.Sen2TCbands)\
            .map(Modules_Sen2Utilities.Sen2SpectralIndices).map(addDate)
        # calculate median to be used in zonal statistics generation
        S2_median = Sentinel2_image_collection.median()
        # convert Sentinel-2 data to numpy array
        S2_data, S2bandNames = buildAndSampleStacks(Sentinel2_image_collection, 'S2', point)

        if len(S2_data) == 0:
            print(' *** Error in extracting Sentinel-2 image stack ***')
            failed_points.append([point_ID, 'Sentinel-2 time series - year '+str(year)])
            continue

        # extract Sentinel-1 time series
        ################################
        S1_enhancement_parameters['GEOMETRY'] = point
        S1_enhancement_parameters['START_DATE'] = str(year) + '-01-01'
        S1_enhancement_parameters['STOP_DATE'] = str(year+1) + '-01-01'
        S1 = Modules_GEE_S1_ARD.s1_preproc(S1_enhancement_parameters, verbose=0)
        Sentinel1_image_collection = S1[1].select(Sentinel1_bands).sort('system:time_start').map(addDate)
        S1_original_scene_list = Sentinel1_image_collection.aggregate_array('system:index').getInfo()
        # Calculate median to be used in zonal statistics generation
        S1_median = Sentinel1_image_collection.median()
        # convert Sentinel-1 data to numpy array
        S1_data, S1bandNames = buildAndSampleStacks(Sentinel1_image_collection, 'S1', point)

        if len(S1_data) == 0:
            print(' *** Error in extracting Sentinel-1 image stack ***')
            failed_points.append([point_ID, 'Sentinel-1 time series - year '+str(year)])
            continue

        # calculate zonal and GLCM statistics based on yearly median images
        ###################################################################
        median_image = S2_median.addBands(S1_median)
        zonalStat_image = ee.Image(0)       # will be dropped later
        reducers = ee.Reducer.min() \
            .combine(reducer2=ee.Reducer.max(), sharedInputs=True) \
            .combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)
        for b in spatial_zonalStat_bands:
            b_stat_3x3 = median_image.select(b)\
                .reduceNeighborhood(reducer=reducers, kernel=ee.Kernel.square(radius=1, units='pixels')) \
                .rename([b+'_min_3x3', b+'_max_3x3', b+'_stdev_3x3'])
            b_stat_5x5 = median_image.select(b)\
                .reduceNeighborhood(reducer=reducers, kernel=ee.Kernel.square(radius=2, units='pixels')) \
                .rename([b+'_min_5x5', b+'_max_5x5', b+'_stdev_5x5'])
            zonalStat_image = zonalStat_image.addBands(b_stat_3x3).addBands(b_stat_5x5)
        zonalStat_image = zonalStat_image.slice(1)
        # add GLCM texture information to yearly image
        if S1_GLCM_band != None:
            zonalStat_image = zonalStat_image.addBands(gen_GLCM(median_image.select(S1_GLCM_band),
                                                          S1_GLCM_band, S1_GLCM_q_level, S1_GLCM_input_min,
                                                          S1_GLCM_input_max, S1_GLCM_features))
        if S2_GLCM_band != None:
            zonalStat_image = zonalStat_image.addBands(gen_GLCM(median_image.select(S2_GLCM_band),
                                                          S2_GLCM_band, S2_GLCM_q_level, S2_GLCM_input_min,
                                                          S2_GLCM_input_max, S2_GLCM_features))
        # sample resulting images
        try:
            zonalStatBandNames = zonalStat_image.bandNames().getInfo()
            zonalStat_data = zonalStat_image.toArray().sample(region=point, scale=scale).getInfo()
            zonalStat_data = np.array(zonalStat_data['features'][0]['properties']['array']).tolist()
        except:
            print(' *** Error in extracting zonal statistics data - year'+str(year))
            failed_points.append([point_ID,'zonalStat data - year'+str(year)])
            continue

        # extract summary yearly climate data from TerraClim dataset
        ############################################################
        terraData = terraClim.filterDate(str(year)+'-01-01', str(year+1)+'-01-01')
        rain_stats = terraData.select('pr').toBands()
        temp_mins = terraData.select('tmmn').toBands()
        temp_maxs = terraData.select('tmmx').toBands()
        rain_stats = rain_stats.select(rain_stats.bandNames(), ['rain_' + x for x in monthNames])
        temp_mins = temp_mins.select(temp_mins.bandNames(), ['tempMin_' + x for x in monthNames])
        temp_maxs = temp_maxs.select(temp_maxs.bandNames(), ['tempMax_' + x for x in monthNames])
        image = rain_stats.addBands(temp_mins).addBands(temp_maxs)
        try:
            terraClim_data = image.toArray().sample(region=point, scale=scale).getInfo()
            terraClim_data = np.array(terraClim_data['features'][0]['properties']['array']).tolist()
        except:
            print(' *** Error in extracting terraClim data - year'+str(year))
            failed_points.append([point_ID,'terraClim data - year'+str(year)])
            continue

        # extract water percentage and night time light data
        ####################################################
        water_band = S2bandNames.index('mndwi')
        water_flag = S2_data[:, water_band] > 0
        # calculate water observation percentage
        water_percentage = sum(water_flag) / len(water_flag)
        # build night time light data for the current year and sample the monthly sequence for current point
        ntl = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG")\
            .filterDate(str(year)+'-01-01',str(year+1)+'-01-01').toArray()
        try:
            ntl_data = ntl.sample(region=point, scale=scale).getInfo()
            ntl_data = np.array(ntl_data['features'][0]['properties']['array'])
            # drop negative readings
            ntl_data[ntl_data[:,0] < 0, 0] = 0
            # keep only months with a minimum number of clear observations
            keep_ind = np.where(ntl_data[:,1] > min_ntl_obs)[0]
            ntl_data = ntl_data[keep_ind,:]
            # sort based on reading values to calculate their weighted median
            ntl_data = ntl_data[np.argsort(ntl_data[:,0])]
            obs_frq_cumsum = np.cumsum(ntl_data[:,1])
            median_freq = obs_frq_cumsum[-1]/2.0
            median_index = np.searchsorted(obs_frq_cumsum, median_freq)
            if median_freq == obs_frq_cumsum[median_index]:
                ntl_median_value = (ntl_data[median_index, 0] + ntl_data[median_index + 1, 0]) / 2.0
            else:
                ntl_median_value = ntl_data[median_index, 0]
        except:
            print(' *** Error in extracting night-time light data - year'+str(year))
            failed_points.append([point_ID, 'night-time light - year'+str(year)])
            continue

        # wrap all data except S1 and S2 into 'annual' data
        if np.any(np.isnan(zonalStat_data)) or np.any(np.isnan(terraClim_data)) or \
                np.isnan(ntl_median_value) or np.isnan(water_percentage):
            annualData = np.nan
        else:
            annualData = zonalStat_data + [ntl_median_value, water_percentage, np.sum(terraClim_data[0:12]),
                                               np.min(terraClim_data[12:24]), np.max(terraClim_data[24:36])]
            annualBandNames = zonalStatBandNames + ['ntl_data', 'water_percentage', 'total_year_rain', 'min_year_temperature',
                                                       'max_year_temperature']

        # append yearly data records to corresponding variables
        variable_data = variable_data + [{'year':year,
                                          'Sentinel2_data': S2_data,
                                          'Sentinel2_original_scene_list': S2_original_scene_list,
                                          'Sentinel2_filtered_scene_list': S2_filtered_scene_list,
                                          'Sentinel1_data': S1_data,
                                          'Sentinel1_original_scene_list': S1_original_scene_list,
                                          'annual_data': np.array(annualData)
                                      }]

    # 2. Generate static features
    ##############################

    print(' - generate static data structures')
    staticBandNames = ['bio_'+str(x).zfill(2) for x in range(1,20)] + \
                                ['elevation', 'slope', 'aspect', 'CHILI_ind', 'soil_data', 'ECO_ID']

    # extract worldclim bio-climate variables
    wclim_data = wclim.sample(region=point, scale=scale).getInfo()
    try:
        d = wclim_data['features'][0]['properties']
        wclim_data = [d['bio_'+str(i).zfill(2)] for i in range(1,20)]
    except:
        print(' *** Error in extracting climate data ***')
        failed_points.append([point_ID, 'climate data'])
        continue
    static_data = wclim_data

    # extract topography data
    topo_data = terrainBands.sample(region=point, scale=scale).getInfo()
    try:
        d = topo_data['features'][0]['properties']
        terrain_data = [d['elevation'], d['slope'], d['aspect']]
    except:
        print(' *** Error in extracting terrain data ***')
        failed_points.append([point_ID, 'terrain data'])
        continue
    static_data = static_data + terrain_data

    # extract chili index
    chili_data = CHILI_index.sample(region=point, scale=scale).getInfo()
    try:
        d = chili_data['features'][0]['properties']
        chili_data = [d['constant']]
    except:
        chili_data = [-1]
    static_data = static_data + chili_data

    # extract soil data
    soil_data = iSDAsoil.sample(region=point, scale=scale).getInfo()
    try:
        d = soil_data['features'][0]['properties']
        soil_data = [d['texture_0_20']]
    except:
        soil_data = [-1]
    static_data = static_data + soil_data

    # extract ecoregion data
    eco_data = ecoBand.sample(region=point, scale=scale).getInfo()
    try:
        eco_data = [eco_data['features'][0]['properties']['first']]
    except:
        eco_data = [-1]
    static_data = static_data + eco_data

    # all yearly and static data to the point data structure
    point_data = point_data + [{'point_ID': point_ID,
                                'coordinates': [lon, lat],
                                'variable_data': variable_data,
                                'fixed_data': {'static_data': np.array(static_data)}
                                }]

# generating output data file. It consists of a .txt file which includes data extraction parameters and a binary
# pickle file which includes point_data variable.
print('*** dumping data to file',out_file_path+out_file_name+'.p ***')
pickle.dump(point_data, open(out_file_path+out_file_name+'.p', 'wb'))
# saving configuration information in a .txt file
f = open(out_file_path + out_file_name + '.txt', 'w')
f.write('Generator script: makeSampleData_V{}.py\n'.format(script_version))
f.write('GEE dataset name: {}\n\n'.format(GEEdatasetName))
f.write('Data configuration:\n')
f.write('-------------------\n')
f.write('Number of points: {}\n'.format(len(point_data)))
f.write('Start year = {}, end_year = {}\n'.format(start_year, end_year))
f.write('Spatial zonal statistics bands = {}\n'.format(spatial_zonalStat_bands))
if S1_GLCM_band != None:
    f.write('Sentinel-1 GLCM base band: {}, Quantization level: {}, Scaling min/max values: {}/{}\n'
            .format(S1_GLCM_band, S1_GLCM_q_level, S1_GLCM_input_min, S1_GLCM_input_max))
    f.write('Sentinel-1 GLCM features : \n')
    for features in S1_GLCM_features:
        f.write('{}\n'.format(features))
if S2_GLCM_band != None:
    f.write('Sentinel-2 GLCM base band: {}, Quantization level: {}, Scaling min/max values: {}/{}\n'
            .format(S2_GLCM_band, S2_GLCM_q_level, S2_GLCM_input_min, S2_GLCM_input_max))
    f.write('Sentinel-2 GLCM features : \n')
    for features in S2_GLCM_features:
        f.write('{}\n'.format(features))
f.write('Minimum nigh-time light observations per month = {}\n'.format(min_ntl_obs))
f.write('Final output feature names:\n - Sentinel-2 bands: {}\n - Sentinel-1 bands: {}\n - Annual bands: {}\n '
        '- Static bands: {}\n'.format(S2bandNames, S1bandNames, annualBandNames, staticBandNames))
f.write('Total processing time: {:.0f} sec\n'.format(time.time() - start_time))
f.write('Points failed to generate data:\n{}'.format(failed_points))
f.close()
