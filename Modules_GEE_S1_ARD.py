###########################################################################################################
# This script is the result of converting all individual scripts in users/adugnagrima/gee_s1_ard to Python
# and combine them in one file. As shown in the MAIN section, you need to define the parameteres and then
# call function s1_preproc. It will return a list of two image collections: Original Sentinel-1 image
# collection and corresponding image collection of processed Sentinel-1 images.

# Modification: Polarization filters added to section #select polarization (6/24/2022)
###########################################################################################################
# Shahriar S. Heydari, 1/29/2022
#
import ee
import math
ee.Initialize()

#---------------------------------------------------------------------------#
# Linear to db scale
#---------------------------------------------------------------------------#

#* Convert backscatter from linear to dB. #
def lin_to_db(image):
  bandNames = image.bandNames().remove('angle')
  db = ee.Image.constant(10).multiply(image.select(bandNames).log10()).rename(bandNames)
  return image.addBands(db, None, True)


#* Convert backscatter from linear to dB. #
def db_to_lin(image):
  bandNames = image.bandNames().remove('angle')
  lin = ee.Image.constant(10).pow(image.select(bandNames).divide(10)).rename(bandNames)
  return image.addBands(lin, None, True)


#Converts the linear image to db by excluding the ratio bands #
def lin_to_db2(image):
  db = ee.Image.constant(10).multiply(image.select(['VV', 'VH']).log10()).rename(['VV', 'VH'])
  return image.addBands(db, None, True)

#---------------------------------------------------------------------------#
# Prepare ratio band for linear image
#---------------------------------------------------------------------------#
def add_ratio_lin(image):
      ratio = image.addBands(image.select('VV').divide(image.select('VH')).rename('VVVH_ratio'))
      return ratio.set('system:time_start', image.get('system:time_start'))

# File: border_noise_correction.js

#---------------------------------------------------------------------------#
# Additional Border Noise Removal
#---------------------------------------------------------------------------#
#* (mask out angles >= 45.23993) #
def maskAngLT452(image):
 ang = image.select(['angle'])
 return image.updateMask(ang.lt(45.23993)).set('system:time_start', image.get('system:time_start'))

#* Function to mask out edges of images using angle.
 # (mask out angles <= 30.63993) #
def maskAngGT30(image):
 ang = image.select(['angle'])
 return image.updateMask(ang.gt(30.63993)).set('system:time_start', image.get('system:time_start'))

#* Remove edges.
 # Source: Andreas Vollrath #
def maskEdge(image):
  mask = image.select(0).unitScale(-25, 5).multiply(255).toByte()#.connectedComponents(ee.Kernel.rectangle(1,1), 100)
  return image.updateMask(mask.select(0)).set('system:time_start', image.get('system:time_start'))

#* Mask edges. This function requires that the input image has one VH or VV band, and an 'angle' bands.  #
def f_mask_edges(image):
  db_img = lin_to_db(image)
  output = maskAngGT30(db_img)
  output = maskAngLT452(output)
  #output = maskEdge(output)
  output = db_to_lin(output)
  return output.set('system:time_start', image.get('system:time_start'))

#---------------------------------------------------------------------------#
# Boxcar filter
#---------------------------------------------------------------------------#
#* Applies boxcar filter on every image in the collection. #
def boxcar(image, KERNEL_SIZE):
    bandNames = image.bandNames().remove('angle')
    # Define a boxcar kernel
    kernel = ee.Kernel.square(radius=(KERNEL_SIZE/2), units='pixels', normalize=True)
    # Apply boxcar
    output = image.select(bandNames).convolve(kernel).rename(bandNames)
    return image.addBands(output, None, True)


#---------------------------------------------------------------------------#
# Lee filter
#---------------------------------------------------------------------------#
#* Lee Filter applied to one image. It is implemented as described in

def leefilter(image,KERNEL_SIZE):
    bandNames = image.bandNames().remove('angle')
    #S1-GRD images are multilooked 5 times in range
    enl = 5
    # Compute the speckle standard deviation
    eta = 1.0/math.sqrt(enl)
    eta = ee.Image.constant(eta)

    # MMSE estimator
    # Neighbourhood mean and variance
    oneImg = ee.Image.constant(1)

    reducers = ee.Reducer.mean().combine(reducer2=ee.Reducer.variance(), sharedInputs=True)

    stats = image.select(bandNames).reduceNeighborhood(reducer=reducers,
                                                       kernel=ee.Kernel.square(KERNEL_SIZE/2, 'pixels'),
                                                       optimization='window')

    meanBand = bandNames.map(lambda bandName: ee.String(bandName).cat('_mean'))
    varBand = bandNames.map(lambda bandName: ee.String(bandName).cat('_variance'))

    z_bar = stats.select(meanBand)
    varz = stats.select(varBand)

    # Estimate weight
    varx = (varz.subtract(z_bar.pow(2).multiply(eta.pow(2)))).divide(oneImg.add(eta.pow(2)))
    b = varx.divide(varz)

    #if b is negative set it to zero
    new_b = b.where(b.lt(0), 0)
    output = oneImg.subtract(new_b).multiply(z_bar.abs()).add(new_b.multiply(image.select(bandNames)))
    output = output.rename(bandNames)
    return image.addBands(output, None, True)

#---------------------------------------------------------------------------#
# GAMMA MAP filter
#---------------------------------------------------------------------------#
#* Gamma Maximum a-posterior Filter applied to one image. It is implemented as described in

def gammamap(image,KERNEL_SIZE):
    enl = 5
    bandNames = image.bandNames().remove('angle')
    #Neighbourhood stats
    reducers = ee.Reducer.mean().combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)
    stats = image.select(bandNames).reduceNeighborhood(reducer=reducers,
                                                       kernel=ee.Kernel.square(KERNEL_SIZE/2, 'pixels'),
                                                       optimization='window')


    meanBand = bandNames.map(lambda bandName: ee.String(bandName).cat('_mean'))
    stdDevBand = bandNames.map(lambda bandName: ee.String(bandName).cat('_stdDev'))

    z = stats.select(meanBand)
    sigz = stats.select(stdDevBand)

    # local observed coefficient of variation
    ci = sigz.divide(z)
    # noise coefficient of variation (or noise sigma)
    cu = 1.0/math.sqrt(enl)
    # threshold for the observed coefficient of variation
    cmax = math.sqrt(2.0) * cu

    cu = ee.Image.constant(cu)
    cmax = ee.Image.constant(cmax)
    enlImg = ee.Image.constant(enl)
    oneImg = ee.Image.constant(1)
    twoImg = ee.Image.constant(2)

    alpha = oneImg.add(cu.pow(2)).divide(ci.pow(2).subtract(cu.pow(2)))

    #Implements the Gamma MAP filter described in equation 11 in Lopez et al. 1990
    q = image.select(bandNames).expression("z**2 * (z * alpha - enl - 1)**2 + 4 * alpha * enl * b() * z", {'z': z, 'alpha': alpha, 'enl': enl})
    rHat = z.multiply(alpha.subtract(enlImg).subtract(oneImg)).add(q.sqrt()).divide(twoImg.multiply(alpha))

    #if ci <= cu then its a homogenous region ->> boxcar filter
    zHat = (z.updateMask(ci.lte(cu))).rename(bandNames)
    #if cmax > ci > cu then its a textured medium ->> apply Gamma MAP filter
    rHat = (rHat.updateMask(ci.gt(cu)).updateMask(ci.lt(cmax))).rename(bandNames)
    #if ci>=cmax then its strong signal ->> retain
    x = image.select(bandNames).updateMask(ci.gte(cmax)).rename(bandNames)

    # Merge
    output = ee.ImageCollection([zHat,rHat,x]).sum()
    return image.addBands(output, None, True)

#---------------------------------------------------------------------------#
# Refined Lee filter
#---------------------------------------------------------------------------#
#* This filter is modified from the implementation by Guido Lemoine
 # Source: Lemoine et al.; https:#code.earthengine.google.com/5d1ed0a0f0417f098fdfd2fa137c3d0c #

def refinedLee(image):

    bandNames = image.bandNames().remove('angle')

    def int_function(b):
        img = image.select([b])

        # img must be linear, i.e. not in dB!
        # Set up 3x3 kernels
        weights3 = ee.List.repeat(ee.List.repeat(1,3),3)
        kernel3 = ee.Kernel.fixed(3,3, weights3, 1, 1, False)

        mean3 = img.reduceNeighborhood(ee.Reducer.mean(), kernel3)
        variance3 = img.reduceNeighborhood(ee.Reducer.variance(), kernel3)

        # Use a sample of the 3x3 windows inside a 7x7 windows to determine gradients and directions
        sample_weights = ee.List([[0,0,0,0,0,0,0], [0,1,0,1,0,1,0],[0,0,0,0,0,0,0], [0,1,0,1,0,1,0], [0,0,0,0,0,0,0], [0,1,0,1,0,1,0],[0,0,0,0,0,0,0]])

        sample_kernel = ee.Kernel.fixed(7,7, sample_weights, 3,3, False)

        # Calculate mean and variance for the sampled windows and store as 9 bands
        sample_mean = mean3.neighborhoodToBands(sample_kernel)
        sample_var = variance3.neighborhoodToBands(sample_kernel)

        # Determine the 4 gradients for the sampled windows
        gradients = sample_mean.select(1).subtract(sample_mean.select(7)).abs()
        gradients = gradients.addBands(sample_mean.select(6).subtract(sample_mean.select(2)).abs())
        gradients = gradients.addBands(sample_mean.select(3).subtract(sample_mean.select(5)).abs())
        gradients = gradients.addBands(sample_mean.select(0).subtract(sample_mean.select(8)).abs())

        # And find the maximum gradient amongst gradient bands
        max_gradient = gradients.reduce(ee.Reducer.max())

        # Create a mask for band pixels that are the maximum gradient
        gradmask = gradients.eq(max_gradient)

        # duplicate gradmask bands: each gradient represents 2 directions
        gradmask = gradmask.addBands(gradmask)

        # Determine the 8 directions
        directions = sample_mean.select(1).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(7))).multiply(1)
        directions = directions.addBands(sample_mean.select(6).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(2))).multiply(2))
        directions = directions.addBands(sample_mean.select(3).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(5))).multiply(3))
        directions = directions.addBands(sample_mean.select(0).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(8))).multiply(4))
        # The next 4 are the not() of the previous 4
        directions = directions.addBands(directions.select(0).Not().multiply(5))
        directions = directions.addBands(directions.select(1).Not().multiply(6))
        directions = directions.addBands(directions.select(2).Not().multiply(7))
        directions = directions.addBands(directions.select(3).Not().multiply(8))

        # Mask all values that are not 1-8
        directions = directions.updateMask(gradmask)

        # "collapse" the stack into a singe band image (due to masking, each pixel has just one value (1-8) in it's directional band, and is otherwise masked)
        directions = directions.reduce(ee.Reducer.sum())

        #pal = ['ffffff','ff0000','ffff00', '00ff00', '00ffff', '0000ff', 'ff00ff', '000000']
        #Map.addLayer(directions.reduce(ee.Reducer.sum()), {min:1, max:8, palette: pal}, 'Directions', False)

        sample_stats = sample_var.divide(sample_mean.multiply(sample_mean))

        # Calculate localNoiseVariance
        sigmaV = sample_stats.toArray().arraySort().arraySlice(0,0,5).arrayReduce(ee.Reducer.mean(), [0])

        # Set up the 7*7 kernels for directional statistics
        rect_weights = ee.List.repeat(ee.List.repeat(0,7),3).cat(ee.List.repeat(ee.List.repeat(1,7),4))

        diag_weights = ee.List([[1,0,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,1,0,0,0,0],
          [1,1,1,1,0,0,0], [1,1,1,1,1,0,0], [1,1,1,1,1,1,0], [1,1,1,1,1,1,1]])

        rect_kernel = ee.Kernel.fixed(7,7, rect_weights, 3, 3, False)
        diag_kernel = ee.Kernel.fixed(7,7, diag_weights, 3, 3, False)

        # Create stacks for mean and variance using the original kernels. Mask with relevant direction.
        dir_mean = img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel).updateMask(directions.eq(1))
        dir_var = img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel).updateMask(directions.eq(1))

        dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel).updateMask(directions.eq(2)))
        dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel).updateMask(directions.eq(2)))

        # and add the bands for rotated kernels
        for i in range(1,4):
          dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)))
          dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)))
          dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)))
          dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)))


        # "collapse" the stack into a single band image (due to masking, each pixel has just one value in it's directional band, and is otherwise masked)
        dir_mean = dir_mean.reduce(ee.Reducer.sum())
        dir_var = dir_var.reduce(ee.Reducer.sum())

        # A finally generate the filtered value
        varX = dir_var.subtract(dir_mean.multiply(dir_mean).multiply(sigmaV)).divide(sigmaV.add(1.0))

        b = varX.divide(dir_var)

        return dir_mean.add(b.multiply(img.subtract(dir_mean))) \
          .arrayProject([0]) \
          .arrayFlatten([['sum']]) \
          .float()

    result = ee.ImageCollection(bandNames.map(int_function)).toBands().rename(bandNames).copyProperties(image)
    return image.addBands(result, None, True)

#---------------------------------------------------------------------------#
# Improved Lee Sigma filter
#---------------------------------------------------------------------------#
#* Implements the improved lee sigma filter to one image.

def leesigma(image,KERNEL_SIZE):
    #parameters
    Tk = ee.Image.constant(7); #number of bright pixels in a 3x3 window
    sigma = 0.9
    enl = 4
    target_kernel = 3
    bandNames = image.bandNames().remove('angle')

    #compute the 98 percentile intensity
    z98 = image.select(bandNames).reduceRegion(reducer=ee.Reducer.percentile([98]), geometry=image.geometry(),
                                               scale=10, maxPixels=1e13).toImage()

    #select the strong scatterers to retain
    brightPixel = image.select(bandNames).gte(z98)
    K = brightPixel.reduceNeighborhood(reducer=ee.Reducer.countDistinctNonNull(),
                                       kernel=ee.Kernel.square((target_kernel/2) ,'pixels'))
    retainPixel = K.gte(Tk)

    #compute the a-priori mean within a 3x3 local window
    #original noise standard deviation
    eta = 1.0/math.sqrt(enl)
    eta = ee.Image.constant(eta)
    #MMSE applied to estimate the a-priori mean
    reducers = ee.Reducer.mean().combine(reducer2=ee.Reducer.variance(), sharedInputs=True)
    stats = image.select(bandNames).reduceNeighborhood(reducer=reducers,
                                                       kernel=ee.Kernel.square(target_kernel/2, 'pixels'),
                                                       optimization='window')
    meanBand = bandNames.map(lambda bandName: ee.String(bandName).cat('_mean'))
    varBand = bandNames.map(lambda bandName: ee.String(bandName).cat('_variance'))

    z_bar = stats.select(meanBand)
    varz = stats.select(varBand)

    oneImg = ee.Image.constant(1)
    varx = (varz.subtract(z_bar.abs().pow(2).multiply(eta.pow(2)))).divide(oneImg.add(eta.pow(2)))
    b = varx.divide(varz)
    xTilde = oneImg.subtract(b).multiply(z_bar.abs()).add(b.multiply(image.select(bandNames)))

    #step 3: compute the sigma range
    # Lookup table (J.S.Lee et al 2009) for range and eta values for intensity (only 4 look is shown here)
    LUT = ee.Dictionary({'0.5': ee.Dictionary({'I1': 0.694,'I2': 1.385,'eta': 0.1921}),
                             '0.6': ee.Dictionary({'I1': 0.630,'I2': 1.495,'eta': 0.2348}),
                             '0.7': ee.Dictionary({'I1': 0.560,'I2': 1.627,'eta': 0.2825}),
                             '0.8': ee.Dictionary({'I1': 0.480,'I2': 1.804,'eta': 0.3354}),
                             '0.9': ee.Dictionary({'I1': 0.378,'I2': 2.094,'eta': 0.3991}),
                             '0.95': ee.Dictionary({'I1': 0.302,'I2': 2.360,'eta': 0.4391})})

    # extract data from lookup
    sigmaImage = ee.Dictionary(LUT.get(str(sigma))).toImage()
    I1 = sigmaImage.select('I1')
    I2 = sigmaImage.select('I2')
    #new speckle sigma
    nEta = sigmaImage.select('eta')
    #establish the sigma ranges
    I1 = I1.multiply(xTilde)
    I2 = I2.multiply(xTilde)

    #step 3: apply the minimum mean square error (MMSE) filter for pixels in the sigma range
    # MMSE estimator
    mask = image.select(bandNames).gte(I1).Or(image.select(bandNames).lte(I2))
    z = image.select(bandNames).updateMask(mask)

    stats = z.reduceNeighborhood(reducer=reducers, kernel= ee.Kernel.square(KERNEL_SIZE/2, 'pixels'),
                                 optimization='window')

    z_bar = stats.select(meanBand)
    varz = stats.select(varBand)

    varx = (varz.subtract(z_bar.abs().pow(2).multiply(nEta.pow(2)))).divide(oneImg.add(nEta.pow(2)))
    b = varx.divide(varz)
    #if b is negative set it to zero
    new_b = b.where(b.lt(0), 0)
    xHat = oneImg.subtract(new_b).multiply(z_bar.abs()).add(new_b.multiply(z))

    # remove the applied masks and merge the retained pixels and the filtered pixels
    xHat = image.select(bandNames).updateMask(retainPixel).unmask(xHat)
    output = ee.Image(xHat).rename(bandNames)
    return image.addBands(output, None, True)


#---------------------------------------------------------------------------#
# 4. Mono-temporal speckle filter
#---------------------------------------------------------------------------#

#* Mono-temporal speckle Filter   #
def MonoTemporal_Filter(coll,KERNEL_SIZE,SPECKLE_FILTER):

    def _filter(image):

        if (SPECKLE_FILTER=='BOXCAR'):
            _filtered = boxcar(image, KERNEL_SIZE)
        elif (SPECKLE_FILTER=='LEE'):
            _filtered = leefilter(image, KERNEL_SIZE)
        elif (SPECKLE_FILTER=='GAMMA MAP'):
            _filtered = gammamap(image, KERNEL_SIZE)
        elif (SPECKLE_FILTER=='REFINED LEE'):
            _filtered = refinedLee(image)
        elif (SPECKLE_FILTER=='LEE SIGMA'):
            _filtered = leesigma(image, KERNEL_SIZE)
        return _filtered

    return coll.map(_filter)

#---------------------------------------------------------------------------#
# 4. Multi-temporal speckle filter
#---------------------------------------------------------------------------#
# The following Multi-temporal speckle filters are implemented as described in

#* Multi-temporal boxcar Filter.  #
def MultiTemporal_Filter(coll,KERNEL_SIZE, SPECKLE_FILTER,NR_OF_IMAGES):

    def Quegan(image):
    # this function will filter the collection used for the multi-temporal part
    #  'it takes care of':
    #     - same image geometry (i.e relative orbit)
    #     - full overlap of image
    #     - amount of images taken for filtering
    #         -- all before
    #        -- if not enough, images taken after the image to filter are added #

        def setresample(image):
            return image.resample()

        def get_filtered_collection(image):
            # filter collection over are and by relative orbit
            s1_coll = ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT')\
                .filterBounds(image.geometry())\
                .filter(ee.Filter.eq('instrumentMode', 'IW'))\
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', ee.List(image.get('transmitterReceiverPolarisation')).get(-1)))\
                .filter(ee.Filter.Or(
                        ee.Filter.eq('relativeOrbitNumber_stop', image.get('relativeOrbitNumber_stop')),
                        ee.Filter.eq('relativeOrbitNumber_stop', image.get('relativeOrbitNumber_start'))
                )).map(setresample)
            # a function that takes the image and checks for the overlap
            def check_overlap(_image):
              # get all S1 frames from this date intersecting with the image bounds
              s1 = s1_coll.filterDate(_image.date(), _image.date().advance(1, 'day'))
              # intersect those images with the image to filter
              intersect = image.geometry().intersection(s1.geometry().dissolve(), 10)
              # check if intersect is sufficient
              valid_date = ee.Algorithms.If(
                intersect.area(10).divide(image.geometry().area(10)).gt(0.95),
                _image.date().format('YYYY-MM-dd')
              )
              return ee.Feature(None, {'date': valid_date})

            # this function will pick up the acq dates for fully overlapping acquisitions before the image acquistion
            dates_before = s1_coll.filterDate('2014-01-01', image.date().advance(1, 'day'))\
                .sort('system:time_start', False).limit(5*NR_OF_IMAGES)\
                .map(check_overlap).distinct('date').aggregate_array('date')

            # if the images before are not enough, we add images from after the image acquisition
            # this will only be the case at the beginning of S1 mission
            dates = ee.List(ee.Algorithms.If(
                dates_before.size().gte(NR_OF_IMAGES),
                dates_before.slice(0, NR_OF_IMAGES),
                s1_coll
                    .filterDate(image.date(), '2100-01-01')
                    .sort('system:time_start', True).limit(5*NR_OF_IMAGES)
                    .map(check_overlap)
                    .distinct('date')
                    .aggregate_array('date')
                    .cat(dates_before).distinct().sort().slice(0, NR_OF_IMAGES))
            )

            # now we re - filter the collection to get the right acquisitions for multi - temporal filtering
            return ee.ImageCollection(dates
                                      .map(lambda date: s1_coll.filterDate(date, ee.Date(date).advance(1, 'day'))
                                           .toList(s1_coll.size())).flatten())

        # we get our dedicated image collection for that image
        s1 = get_filtered_collection(image)

        bands = image.bandNames().remove('angle')
        s1 = s1.select(bands)

        meanBands = bands.map(lambda bandName: ee.String(bandName).cat('_mean'))
        ratioBands = bands.map(lambda bandName: ee.String(bandName).cat('_ratio'))
        count_img = s1.reduce(ee.Reducer.count())
        #estimate means and ratios
        def inner(image):
            if (SPECKLE_FILTER=='BOXCAR'):
                _filtered = boxcar(image, KERNEL_SIZE).select(bands).rename(meanBands)
            elif (SPECKLE_FILTER=='LEE'):
                _filtered = leefilter(image, KERNEL_SIZE).select(bands).rename(meanBands)
            elif (SPECKLE_FILTER=='GAMMA MAP'):
                _filtered = gammamap(image, KERNEL_SIZE).select(bands).rename(meanBands)
            elif (SPECKLE_FILTER=='REFINED LEE'):
                _filtered = refinedLee(image).select(bands).rename(meanBands)
            elif (SPECKLE_FILTER=='LEE SIGMA'):
                _filtered = leesigma(image, KERNEL_SIZE).select(bands).rename(meanBands)

            _ratio = image.select(bands).divide(_filtered).rename(ratioBands)
            return _filtered.addBands(_ratio)

        #perform Quegans filter
        isum = s1.map(inner).select(ratioBands).reduce(ee.Reducer.sum())
        filter = inner(image).select(meanBands)
        divide = filter.divide(count_img)
        output = divide.multiply(isum).rename(bands)

        return image.addBands(output, None, True)

    return coll.map(Quegan)

#---------------------------------------------------------------------------#
# Terrain Flattening
#---------------------------------------------------------------------------#

def slope_correction(collection, TERRAIN_FLATTENING_MODEL, DEM, TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER):

    ninetyRad = ee.Image.constant(90).multiply(math.pi/180)

    def _volumetric_model_SCF(theta_iRad, alpha_rRad):
        nominator = (ninetyRad.subtract(theta_iRad).add(alpha_rRad)).tan()
        denominator = (ninetyRad.subtract(theta_iRad)).tan()
        return nominator.divide(denominator)


    def _direct_model_SCF(theta_iRad, alpha_rRad, alpha_azRad):
        nominator = (ninetyRad.subtract(theta_iRad)).cos()
        denominator = alpha_azRad.cos().multiply((ninetyRad.subtract(theta_iRad).add(alpha_rRad)).cos())
        return nominator.divide(denominator)


    def _erode(image, distance):
        d = (image.Not().unmask(1).fastDistanceTransform(30).sqrt().multiply(ee.Image.pixelArea().sqrt()))

        return image.updateMask(d.gt(distance))


    def _masking(alpha_rRad, theta_iRad, buffer):
        layover = alpha_rRad.lt(theta_iRad).rename('layover')
        # shadow
        shadow = alpha_rRad.gt(ee.Image.constant(-1).multiply(ninetyRad.subtract(theta_iRad))).rename('shadow')
        # combine layover and shadow
        mask = layover.And(shadow)
        # add buffer to final mask
        if (buffer > 0):
            mask = _erode(mask, buffer)
        return mask.rename('no_data_mask')

    def _correct(image):
        bandNames = image.bandNames()
        # get the image geometry and projection
        geom = image.geometry()
        proj = image.select(1).projection()

        #elevation = DEM.reproject(proj).clip(geom)

        elevation = DEM.resample('bilinear').reproject(crs=proj, scale=10).clip(geom)

        # calculate the look direction
        heading = (ee.Terrain.aspect(image.select('angle'))
                   .reduceRegion(ee.Reducer.mean(),image.geometry(),1000)
                   .get('aspect'))

        heading = ee.Algorithms.If(ee.Number(heading).gt(180),ee.Number(heading).subtract(360),ee.Number(heading))

        # the numbering follows the article chapters
        # 2.1.1 Radar geometry
        theta_iRad = image.select('angle').multiply(math.pi/180)
        phi_iRad = ee.Image.constant(heading).multiply(math.pi/180)

        # 2.1.2 Terrain geometry
        #slope
        alpha_sRad = ee.Terrain.slope(elevation).select('slope').multiply(math.pi / 180)

        # aspect (-180 to 180)
        aspect = ee.Terrain.aspect(elevation).select('aspect').clip(geom)

        # we need to subtract 360 degree from all values above 180 degree
        aspect_minus = aspect.updateMask(aspect.gt(180)).subtract(360)

        # we fill the aspect layer with the subtracted values from aspect_minus
        phi_sRad = aspect.updateMask(aspect.lte(180)).unmask().add(aspect_minus.unmask()).multiply(-1).multiply(math.pi / 180)

        # we get the height, for export
        #height = DEM.reproject(proj).clip(geom)

        # 2.1.3 Model geometry
        #reduce to 3 angle
        phi_rRad = phi_iRad.subtract(phi_sRad)

        # slope steepness in range (eq. 2)
        alpha_rRad = (alpha_sRad.tan().multiply(phi_rRad.cos())).atan()

        # slope steepness in azimuth (eq 3)
        alpha_azRad = (alpha_sRad.tan().multiply(phi_rRad.sin())).atan()

        # local incidence angle (eq. 4)
        theta_liaRad = (alpha_azRad.cos().multiply((theta_iRad.subtract(alpha_rRad)).cos())).acos()
        theta_liaDeg = theta_liaRad.multiply(180/math.pi)

        # 2.2
        # Gamma_nought
        gamma0 = image.divide(theta_iRad.cos())

        if (TERRAIN_FLATTENING_MODEL == 'VOLUME'):
            scf = _volumetric_model_SCF(theta_iRad, alpha_rRad)

        if (TERRAIN_FLATTENING_MODEL == 'DIRECT'):
            scf = _direct_model_SCF(theta_iRad, alpha_rRad, alpha_azRad)
        # apply model for Gamm0
        gamma0_flat = gamma0.multiply(scf)

        # get Layover/Shadow mask
        mask = _masking(alpha_rRad, theta_iRad, TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER)

        output = gamma0_flat.mask(mask).rename(bandNames).copyProperties(image)

        output = ee.Image(output).addBands(image.select('angle'),None,True)

        return output.set('system:time_start', image.get('system:time_start'))

    return collection.map(_correct)

def s1_preproc(params, verbose=1):

    #***********************
    # 0. CHECK PARAMETERS
    ########################
    if (params['ORBIT'] == None):
      params['ORBIT'] = 'BOTH'
    if (params['SPECKLE_FILTER'] == None):
      params['SPECKLE_FILTER'] = "GAMMA MAP"
    if (params['SPECKLE_FILTER_KERNEL_SIZE'] == None):
      params['SPECKLE_FILTER_KERNEL_SIZE'] = 7
    if (params['TERRAIN_FLATTENING_MODEL'] == None):
      params['TERRAIN_FLATTENING_MODEL'] = 'VOLUME'
    if (params['TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER'] == None):
      params['TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER'] = 0
    if (params['FORMAT'] == None):
      params['FORMAT'] = 'DB'
    if (params['DEM'] == None):
      params['DEM'] = ee.Image('USGS/SRTMGL1_003')
    if (params['POLARIZATION'] == None):
      params['POLARIZATION'] = 'VVVH'
    if (params['APPLY_ADDITIONAL_BORDER_NOISE_CORRECTION'] == None):
      params['APPLY_ADDITIONAL_BORDER_NOISE_CORRECTION'] = True
    if (params['APPLY_TERRAIN_FLATTENING'] == None):
      params['APPLY_TERRAIN_FLATTTENING'] = True
    if (params['APPLY_SPECKLE_FILTERING'] == None):
      params['APPLY_SPECKLE_FILTERING'] = True
    if (params['SPECKLE_FILTER_FRAMEWORK'] == None):
      params['SPECKLE_FILTER_FRAMEWORK'] = 'MULTI'

    #***********************
    # 1. Data Selection
    ########################

    # Select S1 GRD ImageCollection
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT')\
        .filter(ee.Filter.eq('instrumentMode', 'IW'))\
        .filter(ee.Filter.eq('resolution_meters', 10))\
        .filterDate(params['START_DATE'], params['STOP_DATE'])\
        .filterBounds(params['GEOMETRY'])

    #select orbit
    if (params['ORBIT'] != 'BOTH'):
        s1 = s1.filter(ee.Filter.eq('orbitProperties_pass', params['ORBIT']))

    #select polarization
    if (params['POLARIZATION'] =='VV'):
        s1 = s1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).select(['VV','angle'])
    elif (params['POLARIZATION'] =='VH'):
        s1 = s1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')).select(['VH','angle'])
    elif (params['POLARIZATION'] =='VVVH'):
        s1 = s1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', ['VV', 'VH'])).select(['VV', 'VH', 'angle'])

    if verbose:
        print('Number of images in Sentinel-1 collection: ', s1.size().getInfo())

    #***********************************
    # 2. Additional Border Noise Correction
    ############################## ######

    if (params['APPLY_ADDITIONAL_BORDER_NOISE_CORRECTION']):
        s1_1 = s1.map(f_mask_edges)
        if verbose:
            print('ADDITIONAL BORDER NOISE CORRECTION COMPLETED')
    else:
        s1_1 = s1

    #************************
    # 3. Speckle Filtering
    #########################
    if (params['APPLY_SPECKLE_FILTERING']):
        if (params['SPECKLE_FILTER_FRAMEWORK'] == 'MONO'):
            s1_1 = ee.ImageCollection(MonoTemporal_Filter(s1_1, params['SPECKLE_FILTER_KERNEL_SIZE'],
                                                           params['SPECKLE_FILTER']))
            if verbose:
                print('MONO-TEMPORAL SPECKLE FILTERING COMPLETED')
        else:
            s1_1 = ee.ImageCollection(MultiTemporal_Filter(s1_1, params['SPECKLE_FILTER_KERNEL_SIZE'],
                                                            params['SPECKLE_FILTER'],
                                                            params['SPECKLE_FILTER_NR_OF_IMAGES']))
            if verbose:
                print('MULTI-TEMPORAL SPECKLE FILTERING COMPLETED')

    #**************************************
    # 4. Radiometric Terrain Normalization
    ########################################

    if (params['APPLY_TERRAIN_FLATTENING']):
      s1_1 = ee.ImageCollection(slope_correction(s1_1, params['TERRAIN_FLATTENING_MODEL'],
                                                  params['DEM'],
                                                  params['TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER']))
      if verbose:
          print('RADIOMETRIC TERRAIN NORMALIZATION COMPLETED')

    # Clip to roi(input)
    if (params['CLIP_TO_ROI']):
        s1 = s1.map(lambda image: image.clip(params['GEOMETRY']))
        s1_1 = s1_1.map(lambda image: image.clip(params['GEOMETRY']))

    if (params['FORMAT'] == 'DB'):
        s1 = s1.map(lin_to_db)
        s1_1 = s1_1.map(lin_to_db)

    return [s1, s1_1]


#---------------------------------------------------------------------------#
# MAIN
#---------------------------------------------------------------------------#
if __name__ == '__main__':
    parameter = {#1. Data Selection
                  'START_DATE': "2016-06-01",
                  'STOP_DATE': "2016-09-01",
                  'POLARIZATION':'VVVH',
                  'ORBIT' : 'BOTH',
                  'GEOMETRY': ee.Geometry.Polygon([[[104.8,11.36], [105.16,11.36], [105.16,11.61], [104.8,11.61], [104.8,11.36]]]),
                  #2. Additional Border noise correction
                  'APPLY_ADDITIONAL_BORDER_NOISE_CORRECTION': True,
                  #3.Speckle filter
                  'APPLY_SPECKLE_FILTERING': True,
                  'SPECKLE_FILTER_FRAMEWORK': 'MULTI',
                  'SPECKLE_FILTER': 'LEE',
                  'SPECKLE_FILTER_KERNEL_SIZE': 15,
                  'SPECKLE_FILTER_NR_OF_IMAGES': 10,
                  #4. Radiometric terrain normalization
                  'APPLY_TERRAIN_FLATTENING': True,
                  'DEM': ee.Image('USGS/SRTMGL1_003'),
                  'TERRAIN_FLATTENING_MODEL': 'VOLUME',
                  'TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER': 0,
                  #5. Output
                  'FORMAT' : 'DB',
                  'CLIP_TO_ROI': False,
                  'SAVE_ASSETS': False
    }

    #---------------------------------------------------------------------------#
    # DO THE JOB
    #---------------------------------------------------------------------------#

    #Preprocess the S1 collection
    s1_preprocess = s1_preproc(parameter, verbose=1)

    s1 = s1_preprocess[0]
    s1_processed = s1_preprocess[1]

    image1 = s1.first()
    image2 = s1_processed.first()
    #
    # task = ee.batch.Export.image.toAsset(
    #     image=image1,
    #     description='test_s1',
    #     assetId='users/sheydari/test_s1',
    #     scale=10,
    #     region=parameter['GEOMETRY'].getInfo()['coordinates'],
    #     maxPixels=1e13
    # )
    # task.start()

    # task = ee.batch.Export.image.toAsset(
    #     image=image2,
    #     description='test_s1_processed',
    #     assetId='users/sheydari/test_s1_processed',
    #     scale=10,
    #     region=parameter['GEOMETRY'].getInfo()['coordinates'],
    #     maxPixels=1e13
    # )
    # task.start()

    geometry2 = ee.FeatureCollection([ee.Feature(
        ee.Geometry.Point([104.98274658203123, 11.490526532857375]),
        {
            "system:index": "0"
        })])

    data = image2.sample(region=geometry2, scale=10).getInfo()
    print(data['features'][0]['properties'])