import streamlit as st
import streamlit.components.v1 as components
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
import matplotlib.pyplot as plt
import geemap.foliumap as geemap
from streamlit_folium import folium_static
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve, auc, accuracy_score
import ee 
import os
import numpy as np
import pandas as pd
# import gdal
import tempfile
import rasterio

ee.Authenticate()
ee.Initialize()

def S2_PixelExtraction_Export(Dam_Collection, S2, Hydro, selected_datasets):
    def extract_pixels(box):
        imageDate = ee.Date(box.get("Survey_Date"))
        StartDate = imageDate.advance(-6, 'month').format("YYYY-MM-dd")
        EndDate = imageDate.advance(6, 'month').format("YYYY-MM-dd")
        boxArea = box.geometry()
        damId = box.get("id_property")
        DamStatus = box.get('Dam')
        DamDate = box.get('Damdate')

        filteredCollection = S2.filterDate(StartDate, EndDate).filterBounds(boxArea)

        def add_intersection_ratio(image):
            intersection_mask = image.select('S2_Red').neq(0).clip(boxArea)
            intersection_area = intersection_mask.multiply(ee.Image.pixelArea()).reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=boxArea,
                scale=10,
                maxPixels=1e9
            ).get('S2_Red')

            dam_area = boxArea.area(1)
            intersection_ratio = ee.Number(intersection_area).divide(dam_area).multiply(100).round()

            return image.set('intersection_ratio', intersection_ratio)

        filteredCollection = filteredCollection.map(add_intersection_ratio)
        # Apply focal max filter to fill gaps
        
        NHD_paint = ee.Image(0).int().paint(Hydro, 1, 1)

        NHD_kernel = NHD_paint.focal_max(
            kernelType='circle',
            radius=3,
            units='meters',
            iterations=8
        )

        NHD_raster = NHD_kernel.gt(0).clip(boxArea)

        def add_band(image):
            index = image.get("system:index")
            image_date = ee.Date(image.get('system:time_start'))
            image_month = image_date.get('month')
            intersect = image.get('intersection_ratio')

            additional_bands = []

            if selected_datasets.get("CHIRPS Precipitation", False):
                CHIRPS = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(StartDate, EndDate).filterBounds(boxArea).mean().clip(boxArea)
                additional_bands.append(CHIRPS.select('precipitation').rename('CHIRPS_precipitation_2yr_avg'))

            if selected_datasets.get("ECMWF Precipitation", False):
                ECMWF = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_BY_HOUR").filterDate(StartDate, EndDate).filterBounds(boxArea).mean().clip(boxArea)
                additional_bands.append(ECMWF.select("total_precipitation").rename("ECMWF_precipitation_2yr_avg"))

            if selected_datasets.get("Temperature", False):
                additional_bands.append(ee.Image(ECMWF.select('temperature_2m').rename('ECMWF_temperature_2m_2yr_avg')))

            if selected_datasets.get("Surface Runoff", False):
                additional_bands.append(ee.Image(ECMWF.select("surface_runoff").rename("ECMWF_surface_runoff_2yr_avg")))

            if selected_datasets.get("Elevation", False):
                DEM = ee.Image('USGS/3DEP/10m')
                elevation = DEM.select('elevation').rename('Elevation').clip(boxArea)
                additional_bands.append(elevation)

            if selected_datasets.get("Slope", False):
                slope = ee.Terrain.slope(DEM.select('elevation')).rename('Slope').clip(boxArea)
                additional_bands.append(slope)

            if selected_datasets.get("Vegetation", False):
                USGS_Vegetation = ee.Image('USGS/GAP/CONUS/2011').clip(boxArea)
                additional_bands.append(USGS_Vegetation.rename('Vegetation'))

            Full_image = image.addBands(NHD_raster.rename("NHD_Hydro")).addBands(additional_bands).set({
                "First_id": ee.String(damId).cat("_").cat(DamStatus).cat("_S2id:_").cat(index).cat("_").cat(DamDate).cat("_intersect_").cat(intersect),
                "Dam_id": damId,
                "Dam_status": DamStatus,
                "Image_month": image_month
            }).clip(boxArea)

            return Full_image

        filteredCollection = filteredCollection.map(add_band)

        def calculate_cloud_coverage(image):
            cloud = image.select('S2_Binary_cloudMask')
            cloud_stats = cloud.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=image.geometry(),
                scale=10,
                maxPixels=1e9
            )
            clear_coverage_percentage = ee.Number(cloud_stats.get('S2_Binary_cloudMask')).multiply(100).round()
            cloud_coverage_percentage = ee.Number(100).subtract(clear_coverage_percentage)
            return image.set('Cloud_coverage', cloud_coverage_percentage)

        filteredCollection = filteredCollection.map(calculate_cloud_coverage)

        filteredCollection = filteredCollection.filterMetadata('intersection_ratio', 'greater_than', 0.95)

        def get_monthly_least_cloudy_images(Collection):
            months = ee.List.sequence(1, 12)
            def get_month_image(month):
                monthly_images = Collection.filter(ee.Filter.calendarRange(month, month, 'month'))
                return ee.Image(monthly_images.sort('CLOUDY_PIXEL_PERCENTAGE').first())
            return ee.ImageCollection.fromImages(months.map(get_month_image))

        filteredCollection = get_monthly_least_cloudy_images(filteredCollection)

        def addCloud(image):
            cloud = image.get("Cloud_coverage")
            return image.set("Full_id", ee.String(image.get("First_id")).cat("_Cloud_").cat(cloud))

        return filteredCollection.map(addCloud)

    ImageryCollections = Dam_Collection.map(extract_pixels).flatten()
    return ee.ImageCollection(ImageryCollections)

def Sentinel_Only_Export(Dam_Collection, S2):
    def extract_pixels(box):
        imageDate = ee.Date(box.get("Survey_Date"))
        StartDate = imageDate.advance(-6, 'month').format("YYYY-MM-dd")
        EndDate = imageDate.advance(6, 'month').format("YYYY-MM-dd")
        boxArea = box.geometry()
        DateString = box.get("stringID")
        damId = box.get("id_property")
        DamStatus = box.get('Dam')
        DamDate = box.get('Damdate')
        DamGeo = box.get('Point_geo')
        filteredCollection = S2.filterDate(StartDate, EndDate).filterBounds(boxArea)

        def add_band(image):
            index = image.get("system:index")
            image_date = ee.Date(image.get('system:time_start'))
            image_month = image_date.get('month')
            image_year = image_date.get('year')
            cloud = image.get('CLOUDY_PIXEL_PERCENTAGE')
            # intersect = image.get('intersection_ratio')
            
            dataset = ee.Image('USGS/3DEP/10m')
            elevation_select = dataset.select('elevation')
            elevation = ee.Image(elevation_select)
    
            # Extract sample area from elevation
            point_geom = DamGeo
    
            # Extract elevation of dam location
            point_elevation = ee.Number(elevation.sample(point_geom, 10).first().get('elevation'))
            buffered_area = boxArea
            elevation_clipped = elevation.clip(buffered_area)
    
            # Create elevation radius around point to sample from
            point_plus = point_elevation.add(3)
            point_minus = point_elevation.subtract(5)        
            elevation_masked = elevation_clipped.where(elevation_clipped.lt(point_minus), 0).where(elevation_clipped.gt(point_minus), 1).where(elevation_clipped.gt(point_plus), 0)
            elevation_masked2 = elevation_masked.updateMask(elevation_masked.eq(1));
    
            # Add bands, create new "id" property to name the file, and clip the images to the ROI
            # Full_image = image.set("First_id", ee.String(damId).cat("_").cat(DamStatus).cat("_S2id:_").cat(index).cat("_").cat(DamDate).cat("_intersect_").cat(intersect))\
            Full_image = image.set("First_id", ee.String(damId).cat("_").cat(DamStatus).cat("_S2id:_").cat(index).cat("_").cat(DamDate).cat("_intersect_"))\
                .set("Dam_id",damId)\
                .set("Dam_status",DamStatus)\
                .set("Image_month",image_month)\
                .set("Image_year",image_year)\
                .clip(boxArea)
            Full_image2 = Full_image.addBands(elevation_masked2)
            ## maybe add point geo as a property
            
           
            return Full_image2

        filteredCollection = filteredCollection.map(add_band)

        def calculate_cloud_coverage(image):
            cloud = image.select('S2_Binary_cloudMask')
            cloud_stats = cloud.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=image.geometry(),
                scale=10,
                maxPixels=1e9
            )
            clear_coverage_percentage = ee.Number(cloud_stats.get('S2_Binary_cloudMask')).multiply(100).round()
            cloud_coverage_percentage = ee.Number(100).subtract(clear_coverage_percentage)
            return image.set('Cloud_coverage', cloud_coverage_percentage)

        filteredCollection = filteredCollection.map(calculate_cloud_coverage)


        def get_monthly_least_cloudy_images(Collection):
            months = ee.List.sequence(1, 12)
            def get_month_image(month):
                monthly_images = Collection.filter(ee.Filter.calendarRange(month, month, 'month'))
                return ee.Image(monthly_images.sort('CLOUDY_PIXEL_PERCENTAGE').first())
            return ee.ImageCollection.fromImages(months.map(get_month_image))

        filteredCollection = get_monthly_least_cloudy_images(filteredCollection)

        def addCloud(image):
            cloud = image.get("Cloud_coverage")
            return image.set("Full_id", ee.String(image.get("First_id")).cat("_Cloud_").cat(cloud))

        return filteredCollection.map(addCloud)

    ImageryCollections = Dam_Collection.map(extract_pixels).flatten()
    return ee.ImageCollection(ImageryCollections)

def S2_Export_for_visual(Dam_Collection):
    def extract_pixels(box):
        imageDate = ee.Date(box.get("Survey_Date"))
        StartDate = imageDate.advance(-6, 'month').format("YYYY-MM-dd")

        EndDate = imageDate.advance(6, 'month').format("YYYY-MM-dd")

        boxArea = box.geometry()
        DateString = box.get("stringID")
        damId = box.get("id_property")
        DamStatus = box.get('Dam')
        DamDate = box.get('Damdate')
        DamGeo = box.get('Point_geo')
        S2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")

        ## Add band for cloud coverage
        def add_cloud_mask_band(image):
            qa = image.select('QA60')

            # Bits 10 and 11 are clouds and cirrus, respectively.
            cloudBitMask = 1 << 10
            cirrusBitMask = 1 << 11

            # Both flags should be set to zero, indicating clear conditions.
            cloud_mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(
                qa.bitwiseAnd(cirrusBitMask).eq(0)
            )
            # Create a band with values 1 (clear) and 0 (cloudy or cirrus) and convert from byte to Uint16
            cloud_mask_band = cloud_mask.rename('cloudMask').toUint16()

            return image.addBands(cloud_mask_band)

        # Define the dataset
        S2_cloud_band = S2.map(add_cloud_mask_band)


        # Change band names
        oldBandNames = ['B2', 'B3', 'B4', 'B8','cloudMask']
        newBandNames = ['S2_Blue', 'S2_Green', 'S2_Red', 'S2_NIR','S2_Binary_cloudMask']#'S2_NDVI']

        S2_named_bands = S2_cloud_band.map(lambda image: image.select(oldBandNames).rename(newBandNames))

        # Define a function to add the masked image as a band to the images
        def addAcquisitionDate(image):
            date = ee.Date(image.get('system:time_start'))
            return image.set('acquisition_date', date)

        S2_cloud_filter = S2_named_bands.map(addAcquisitionDate)

        filteredCollection = S2_cloud_filter.filterDate(StartDate, EndDate).filterBounds(boxArea)

            
        def add_band(image):
            index = image.get("system:index")
            image_date = ee.Date(image.get('system:time_start'))
            image_month = image_date.get('month')
            image_year = image_date.get('year')
            cloud = image.get('CLOUDY_PIXEL_PERCENTAGE')
            # intersect = image.get('intersection_ratio')
            
            dataset = ee.Image('USGS/3DEP/10m')
            elevation_select = dataset.select('elevation')
            elevation = ee.Image(elevation_select)

            # Extract sample area from elevation
            point_geom = DamGeo

            # Extract elevation of dam location
            point_elevation = ee.Number(elevation.sample(point_geom, 10).first().get('elevation'))
            buffered_area = boxArea
            elevation_clipped = elevation.clip(buffered_area)

            # Create elevation radius around point to sample from
            point_plus = point_elevation.add(3)
            point_minus = point_elevation.subtract(5)        
            elevation_masked = elevation_clipped.where(elevation_clipped.lt(point_minus), 0).where(elevation_clipped.gt(point_minus), 1).where(elevation_clipped.gt(point_plus), 0)
            elevation_masked2 = elevation_masked.updateMask(elevation_masked.eq(1));

            # Add bands, create new "id" property to name the file, and clip the images to the ROI
            # Full_image = image.set("First_id", ee.String(damId).cat("_").cat(DamStatus).cat("_S2id:_").cat(index).cat("_").cat(DamDate).cat("_intersect_").cat(intersect))\
            Full_image = image.set("First_id", ee.String(damId).cat("_").cat(DamStatus).cat("_S2id:_").cat(index).cat("_").cat(DamDate).cat("_intersect_"))\
                .set("Dam_id",damId)\
                .set("Dam_status",DamStatus)\
                .set("Image_month",image_month)\
                .set("Image_year",image_year)\
                .clip(boxArea)
            Full_image2 = Full_image.addBands(elevation_masked2)
            ## maybe add point geo as a property
            
        
            return Full_image2
            
        filteredCollection2 = filteredCollection.map(add_band)

        def calculate_cloud_coverage(image):
            cloud = image.select('S2_Binary_cloudMask')

            # Compute cloud coverage percentage using a simpler approach
            cloud_stats = cloud.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=image.geometry(),
                scale=10,
                maxPixels=1e9
            )
            
            clear_coverage_percentage = ee.Number(cloud_stats.get('S2_Binary_cloudMask')).multiply(100).round()
            cloud_coverage_percentage = ee.Number(100).subtract(clear_coverage_percentage)  # Invert the percentage

            return image.set('Cloud_coverage', cloud_coverage_percentage)


        filteredCloudCollection = filteredCollection2.map(calculate_cloud_coverage)
        # filteredCollection_overlap = filteredCloudCollection.filterMetadata('intersection_ratio', 'greater_than', 0.95)

        
        # Group by month and get the least cloudy image for each month
        def get_monthly_least_cloudy_images(Collection):
            months = ee.List.sequence(1, 12)
            def get_month_image(month):
                monthly_images = Collection.filter(ee.Filter.calendarRange(month, month, 'month'))
                return ee.Image(monthly_images.sort('CLOUDY_PIXEL_PERCENTAGE').first())
            
            monthly_images_list = months.map(get_month_image)
            monthly_images_collection = ee.ImageCollection.fromImages(monthly_images_list)
            return monthly_images_collection
            
        # filteredCollectionBands = get_monthly_least_cloudy_images(filteredCollection_overlap) 
        filteredCollectionBands = get_monthly_least_cloudy_images(filteredCloudCollection) 

        def addCloud(image):
            id = image.get("First_id")
            cloud = image.get("Cloud_coverage")
            Complete_id = image.set("Full_id", ee.String(id).cat("_Cloud_").cat(cloud))
            return Complete_id

        Complete_collection = filteredCollectionBands.map(addCloud)
        
        return Complete_collection 

    ImageryCollections = Dam_Collection.map(extract_pixels).flatten()
    return ee.ImageCollection(ImageryCollections)




def compute_ndvi_mean(image):
    # Select elevation for geometry
    elevation_mask = image.select('elevation')

    # Compute NDVI
    ndvi_mean = image.normalizedDifference(['S2_NIR', 'S2_Red']).rename('NDVI').reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=elevation_mask.geometry(),
        scale=30,
        maxPixels=1e13
    )
    
    # Compute NDWI_Green
    ndwi_green_mean = image.normalizedDifference(['S2_Green', 'S2_NIR']).rename('NDWI_Green').reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=elevation_mask.geometry(),
        scale=30,
        maxPixels=1e13
    )
    
    # Extract month and dam status
    month = image.get('Image_month')
    status = image.get('Dam_status')

    # Combine all metrics and metadata
    combined_metrics = ee.Dictionary({
        'NDVI': ndvi_mean.get('NDVI'),
        'NDWI_Green': ndwi_green_mean.get('NDWI_Green'),
        'Image_month': month,
        'Dam_status': status
    })

    # Return a feature with combined metrics as properties
    return ee.Feature(None, combined_metrics)      


# def S2_PixelExtraction_Export(Dam_Collection,S2,Hydro):
#     def extract_pixels(box):
#         imageDate = ee.Date(box.get("Survey_Date"))
#         StartDate = imageDate.advance(-6, 'month').format("YYYY-MM-dd")
#         EndDate = imageDate.advance(6, 'month').format("YYYY-MM-dd")
#         boxArea = box.geometry()
#         damId = box.get("id_property")
#         DamStatus = box.get('Dam')
#         DamDate = box.get('Damdate')
        
#         filteredCollection = S2.filterDate(StartDate, EndDate).filterBounds(boxArea)
#         def add_intersection_ratio(image):
#             # Calculate intersection mask
#             intersection_mask = image.select('S2_Red').neq(0).clip(boxArea)
            
#             # Compute the area of the intersection
#             intersection_area = intersection_mask.multiply(ee.Image.pixelArea()).reduceRegion(
#                 reducer=ee.Reducer.sum(),
#                 geometry=boxArea,
#                 scale=10,
#                 maxPixels=1e9
#             ).get('S2_Red')
            
#             # Compute the area of the original dam box
#             dam_area = boxArea.area(1)
            
#             # Compute intersection ratio
#             intersection_ratio = ee.Number(intersection_area).divide(dam_area).multiply(100).round()
            
#             return image.set('intersection_ratio', intersection_ratio)
        
#         filteredCollection = filteredCollection.map(add_intersection_ratio)
        
#         def add_band(image):
#             index = image.get("system:index")
#             image_date = ee.Date(image.get('system:time_start'))
#             image_month = image_date.get('month')
#             cloud = image.get('CLOUDY_PIXEL_PERCENTAGE')
#             intersect = image.get('intersection_ratio')
            
#             # # Climate metrics- need to convert to int
            

#             CHIRPS = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(StartDate, EndDate).filterBounds(boxArea).mean().clip(boxArea)
#             CHIRPS_precipitation = ee.Image(CHIRPS.select('precipitation').rename('CHIRPS_precipitation_2yr_avg'))
#             ECMWF = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_BY_HOUR").filterDate(StartDate, EndDate).filterBounds(boxArea).mean().clip(boxArea)
#             ECMWF_precipitation = ee.Image(ECMWF.select("total_precipitation").rename("ECMWF_precipitation_2yr_avg"))
#             temperature = ee.Image(ECMWF.select('temperature_2m').rename('ECMWF_temperature_2m_2yr_avg'))
#             surface_runoff = ee.Image(ECMWF.select("surface_runoff").rename("ECMWF_surface_runoff_2yr_avg"))
#             USGS_Vegetation = ee.Image('USGS/GAP/CONUS/2011')
#             DEM = ee.Image('USGS/3DEP/10m')
#             elevation_select = DEM.select('elevation')
#             elevation = ee.Image(elevation_select)
#             slope_select = ee.Terrain.slope(elevation)
#             slope = ee.Image(slope_select)
               
#             Full_image = image.addBands(Hydro.rename("NHD_Hydro"))\
#                 .addBands(USGS_Vegetation)\
#                 .addBands(slope).addBands(elevation)\
#                 .addBands(ECMWF_precipitation).addBands(temperature).addBands(surface_runoff)\
#                 .set("First_id", ee.String(damId).cat("_").cat(DamStatus).cat("_S2id:_").cat(index).cat("_").cat(DamDate).cat("_intersect_").cat(intersect))\
#                 .set("Dam_id",damId)\
#                 .set("Dam_status",DamStatus)\
#                 .set("Image_month",image_month)\
#                 .clip(boxArea)
#             # .addBands(Stream_order.rename("StreamOrder"))\ 

#             # Add bands, create new "id" property to name the file, and clip the images to the ROI
#             return Full_image
         

#         def calculate_cloud_coverage(image):
#             cloud = image.select('S2_Binary_cloudMask')
        
#             # Compute cloud coverage percentage using a simpler approach
#             cloud_stats = cloud.reduceRegion(
#                 reducer=ee.Reducer.mean(),
#                 geometry=image.geometry(),
#                 scale=10,
#                 maxPixels=1e9
#             )
            
#             clear_coverage_percentage = ee.Number(cloud_stats.get('S2_Binary_cloudMask')).multiply(100).round()
#             cloud_coverage_percentage = ee.Number(100).subtract(clear_coverage_percentage)  # Invert the percentage
        
#             return image.set('Cloud_coverage', cloud_coverage_percentage)
            
#         filteredCollection2 = filteredCollection.map(add_band)
#         filteredCloudCollection = filteredCollection2.map(calculate_cloud_coverage)
#         filteredCollection_overlap = filteredCloudCollection.filterMetadata('intersection_ratio', 'greater_than', 0.95)
       
        
#         # Group by month and get the least cloudy image for each month
#         def get_monthly_least_cloudy_images(Collection):
#             months = ee.List.sequence(1, 12)
#             def get_month_image(month):
#                 monthly_images = Collection.filter(ee.Filter.calendarRange(month, month, 'month'))
#                 return ee.Image(monthly_images.sort('CLOUDY_PIXEL_PERCENTAGE').first())
            
#             monthly_images_list = months.map(get_month_image)
#             monthly_images_collection = ee.ImageCollection.fromImages(monthly_images_list)
#             return monthly_images_collection
            
#         filteredCollectionBands = get_monthly_least_cloudy_images(filteredCollection_overlap) 

#         def addCloud(image):
#             id = image.get("First_id")
#             cloud = image.get("Cloud_coverage")
#             Complete_id = image.set("Full_id", ee.String(id).cat("_Cloud_").cat(cloud))
#             return Complete_id

#         Complete_collection = filteredCollectionBands.map(addCloud)
        
#         return Complete_collection 

#     ImageryCollections = Dam_Collection.map(extract_pixels).flatten()
#     return ee.ImageCollection(ImageryCollections)