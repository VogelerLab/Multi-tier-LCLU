import math
import ee
import numpy as np
import datetime
ee.Initialize()

global CLD_PRB_THRESH #= 30
global NIR_DRK_THRESH #= .15
global CLD_PRJ_DIST #= 10
global BUFFER #= 200

def Sen2CloudMasking(img):
    # source: https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless
    # Note: The original code was based on Sentinel-2 BOA with classification layer. But I want to use
    # it over the output of my Sen2Cor module, which does not produce that classification layer. So one
    # line of code where the classification water layer was used for shadow processing was removed.
    # Also, original image mask was combined with produced mask in cloud filtering procedure (10/17/2022)
    # Shahriar S. Heydari

    # Add cloud component bands.
    cld_prb = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterMetadata('system:index', 'equals', img.get('system:index')).first().select('probability'))

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(CLD_PRB_THRESH)#.rename('cloudMask')
    # Add the cloud probability layer and cloud mask as image bands.
    # img_cloud = img.addBands(ee.Image(is_cloud))

    # Add cloud shadow component bands.
    # Identify dark NIR pixels (potential cloud shadow pixels).
    dark_pixels = img.select('B8').lt(NIR_DRK_THRESH)#.rename('dark_pixels')

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = (is_cloud.directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
        .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
        .select('distance')
        .mask())

    # Identify the intersection of dark pixels with cloud shadow projection.
    is_shadow = cld_proj.multiply(dark_pixels)#.rename('shadows')

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = is_cloud.add(is_shadow).gt(0)

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(BUFFER*2/20)
        .reproject(**{'crs': img.select([0]).projection(), 'scale': 20}))#.rename('cloudAndShadowMask'))

    # Use the final cloud-shadow mask to update the image mask and return it
    return img.updateMask(is_cld_shdw.Not())

def Sen2SpectralIndices(image):
    # custom processing for each Sentinel2 image
    img = image.select(['B2','B3','B4','B8','B11','B12'],['blue','green','red','nir','swir1','swir2'])

    ndvi = img.expression('(b("nir")-b("red"))/(b("nir")+b("red"))').rename('ndvi')

    # MNDWI is used to detect water bodies (positive values are likely water pixels)
    mndwi = img.expression('(b("green")-b("swir1"))/(b("green")+b("swir1"))').rename('mndwi')

    # calculate UCI spectral index (used for separating impervious, bare soil, and vegetation land covers
    # reference: https://www.mdpi.com/2072-4292/13/1/3
    uci = img.expression('(b("blue")-(2.0*b("nir")*b("swir1")/(b("nir")+b("swir1"))))/'
                         '(b("blue")+(2.0*b("nir")*b("swir1")/(b("nir")+b("swir1"))))').rename('uci')

    # calculate a custom wetness index
    wi = img.expression('(b("nir")**2+b("red")**2)**0.5').rename('wi')

    # calculate three other indices that can be helpful for built-up/paved area identification

    # reference: https://www.sciencedirect.com/science/article/pii/S111098231400043X
    bai = img.expression('(b("blue")-b("nir"))/(b("blue")+b("nir"))').rename('bai')

    # reference: https://www.researchgate.net/publication/352929075_Review_of_Spectral_Indices_for_Urban_Remote_Sensing
    nbai = img.expression('(b("swir1")-(b("swir2")/b("green")))/(b("swir1")+(b("swir2")/b("green")))').rename('nbai')
    baei = img.expression('(b("red")+0.3)/(b("green")+b("swir1"))').rename('baei')

    return image.addBands([ndvi, mndwi, uci, wi, bai, nbai, baei])

def Sen2TCbands(image):
    # Tasseled-cap coefficients are calculated from Sentinel-2 TOA six bands as described in Shi and Xu (2019),
    # DOI:10.1109/JSTARS.2019.2938388
    img = image.select(['B2','B3','B4','B8','B11','B12'],['blue','green','red','nir','swir1','swir2'])
    brt_coeffs = ee.Image.constant([0.3510, 0.3813, 0.3437, 0.7196, 0.2396, 0.1949])
    grn_coeffs = ee.Image.constant([-0.3599, -0.3533, -0.4734, 0.6633, 0.0087, -0.2856])
    wet_coeffs = ee.Image.constant([0.2578, 0.2305, 0.0883, 0.1071, -0.7611, -0.5308])
    sum = ee.Reducer.sum()
    brightness = img.multiply(brt_coeffs).reduce(sum)
    greenness = img.multiply(grn_coeffs).reduce(sum)
    wetness = img.multiply(wet_coeffs).reduce(sum)
    angle = (greenness.divide(brightness)).atan().multiply(180/np.pi)
    tc = brightness.addBands(greenness).addBands(wetness).addBands(angle).select([0, 1, 2, 3],
                                                                                 ['tcb', 'tcg', 'tcw', 'tca'])
    return image.addBands(tc)

# if __name__ == '__main__':
#     AOI = ee.Geometry.Point(12.670733, 41.826685)  # ESRIN (ESA Earth Observation Centre)
#     START_DATE = '2018-07-01'
#     END_DATE = '2018-09-01'
#     NIR_DRK_THRESH = NIR_DRK_THRESH * 10000
#     image = (ee.ImageCollection('COPERNICUS/S2')
#         .filterBounds(AOI)
#         .filterDate(START_DATE, END_DATE)).first()
#     image = process_image_Sen2CldMasking(image)
#     info = image.getInfo()['properties']
#     scene_date = datetime.datetime.utcfromtimestamp(info['system:time_start']/1000).strftime("%Y-%m-%d")# i.e. Python uses seconds, EE uses milliseconds
#     assetID = 'users/sheydari/S2_cloudMasked_'+scene_date
#     region = AOI.buffer(5000).bounds().getInfo()['coordinates']
#
#     # # export
#     export = ee.batch.Export.image.toAsset(
#         image=image,
#         description='sentinel2_cloudMasking_export',
#         assetId = assetID,
#         region = region,
#         scale = 10)
#
#     # # uncomment to run the export
#     export.start()
