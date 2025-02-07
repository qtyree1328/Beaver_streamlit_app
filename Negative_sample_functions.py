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

def deduplicate_locations(orig_set):
    """Merge close points, take centroids, and return unique feature collection."""
    bufferDistance = 10  # Adjust based on your criteria
    bufferedPoints = orig_set.map(lambda point: point.buffer(bufferDistance))
    unionOfBuffers = bufferedPoints.union()
    simplifiedUnion = unionOfBuffers.geometry().simplify(bufferDistance / 2)
    centroids = simplifiedUnion.geometries().map(
        lambda geom: ee.Feature(ee.Geometry(geom).centroid())
    )
    return ee.FeatureCollection(centroids)

def prepareHydro(waterway_fc):
    """
    Convert a lines/polygons FeatureCollection (hydro) to a raster image
    for sampling, with 'hydro_mask' band = 1 where waterway is present.
    """
    # Paint the features onto an empty image
    base = ee.Image(0).int()
    hydroRaster = base.paint(waterway_fc, 1, 1)
    
    # Focal max to fill small gaps (optional). Adjust radius/iterations as needed
    filledImage = hydroRaster.focal_max(radius=2, units="meters", iterations=8)
    hydroRaster = filledImage.gt(0).rename(["hydro_mask"])
    
    return hydroRaster

def sampleNegativePoints(positiveDams, hydroRaster, innerRadius, outerRadius, samplingScale):
    """
    Create negative points by:
      1) Buffering positive sites with innerRadius, dissolving them.
      2) Buffering that dissolved geometry again by outerRadius.
      3) Taking the difference (outer minus inner).
      4) Sampling from hydroRaster within that ring, ensuring hydro_mask == 1.
    """
    
    # Buffer each point by innerRadius
    innerBuffers = positiveDams.map(lambda pt: pt.buffer(innerRadius))
    innerDissolved = innerBuffers.geometry().dissolve()
    
    # Buffer the dissolved geometry by outerRadius
    outerBuffer = ee.Feature(innerDissolved.buffer(outerRadius))
    # We want just the ring (outer minus inner)
    ringArea = outerBuffer.geometry().difference(innerDissolved)
    
    # Clip hydroRaster to that ring
    # (Although not strictly necessary if we use region=ringArea)
    clippedHydro = hydroRaster.clip(ringArea)
    
    # Sample the same number of negatives as positives
    numPoints = positiveDams.size()
    
    # Use stratifiedSample, specifying classBand='hydro_mask'
    samples = clippedHydro.stratifiedSample(
        numPoints=numPoints,
        classBand='hydro_mask',
        region=ringArea,
        scale=samplingScale,
        seed=42,
        geometries=True
    )
    
    # Filter only where hydro_mask == 1
    negativePoints = samples.filter(ee.Filter.eq('hydro_mask', 1))
    return negativePoints