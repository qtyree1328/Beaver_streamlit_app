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

def process_Sentinel2_with_cloud_coverage(S2_collection):
    """
    Process Sentinel-2 collection by adding a cloud mask band, renaming bands,
    adding acquisition date, and calculating cloud coverage percentage.
    """
    # Add cloud mask band
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

    # Rename bands
    oldBandNames = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'cloudMask']
    newBandNames = [
        'S2_Aerosols', 'S2_Blue', 'S2_Green', 'S2_Red', 'S2_Red_Edge1',
        'S2_Red_Edge2', 'S2_Red_Edge3', 'S2_NIR', 'S2_Red_Edge4',
        'S2_Water_Vapor', 'S2_SWIR1', 'S2_SWIR2', 'S2_Binary_cloudMask'
    ]

    def rename_bands(image):
        return image.select(oldBandNames).rename(newBandNames)

    # Add acquisition date
    def add_acquisition_date(image):
        date = ee.Date(image.get('system:time_start'))
        return image.set('acquisition_date', date)

    # Process the collection
    processed_collection = (
        S2_collection
        .map(add_cloud_mask_band)
        .map(rename_bands)
        .map(add_acquisition_date)
    )
    
    return processed_collection


