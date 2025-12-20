"""
Fonctions projet de télédétection
Auteur : Djiby Diallo
Cours : Télédétection Approfondissement - Qualité et fouille de données
"""

import numpy as np
import os
from osgeo import gdal, ogr


def create_stack(file_list, output_path):
    """
    Crée un stack multi-temporel à partir d'une liste de fichiers raster.
    L'ordre de file_list doit correspondre à l'ordre temporel.
    """
    if len(file_list) == 0:
        raise ValueError("La liste de fichiers est vide : impossible de créer un stack.")

    vrt_temp = output_path.replace(".tif", ".vrt")

    gdal.BuildVRT(vrt_temp, file_list, separate=True)
    gdal.Translate(output_path, vrt_temp, format="GTiff")

    os.remove(vrt_temp)
    return output_path


def rasterize_vector(vector_path, ref_raster_path, output_path, attribute="strate"):
    """
    Rasterise un shapefile en utilisant un raster de référence.
    """
    ds_ref = gdal.Open(ref_raster_path)
    if ds_ref is None:
        raise IOError("Impossible d'ouvrir le raster de référence.")

    gt = ds_ref.GetGeoTransform()
    proj = ds_ref.GetProjection()
    cols = ds_ref.RasterXSize
    rows = ds_ref.RasterYSize

    driver = gdal.GetDriverByName("GTiff")
    ds_out = driver.Create(output_path, cols, rows, 1, gdal.GDT_Byte)
    ds_out.SetGeoTransform(gt)
    ds_out.SetProjection(proj)

    band = ds_out.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.Fill(0)

    ds_vect = ogr.Open(vector_path)
    layer = ds_vect.GetLayer()

    gdal.RasterizeLayer(
        ds_out,
        [1],
        layer,
        options=[f"ATTRIBUTE={attribute}"]
    )

    ds_out = None
    ds_ref = None
    ds_vect = None

    return output_path


def extract_stats_by_class(image, mask, classes, nodata=-9999):
    """
    Extrait moyenne et écart-type par classe et par date.
    """
    nb_dates = image.shape[2]
    nb_classes = len(classes)

    moyennes = np.zeros((nb_classes, nb_dates))
    ecarts_types = np.zeros((nb_classes, nb_dates))

    for i, classe in enumerate(classes):
        mask_classe = (mask == classe)

        for d in range(nb_dates):
            valeurs = image[:, :, d][mask_classe]
            valeurs = valeurs[valeurs != nodata]

            if valeurs.size > 0:
                moyennes[i, d] = np.mean(valeurs)
                ecarts_types[i, d] = np.std(valeurs)
            else:
                moyennes[i, d] = np.nan
                ecarts_types[i, d] = np.nan

    return moyennes, ecarts_types


def calculate_ari(B03, B05, nodata=-9999):
    """
    Calcule l'indice ARI avec gestion explicite des divisions invalides.
    """
    ARI = np.full(B03.shape, nodata, dtype=np.float32)

    valid_mask = (B03 > 0) & (B05 > 0)

    np.seterr(divide="ignore", invalid="ignore")
    ARI[valid_mask] = (
        (1.0 / B03[valid_mask] - 1.0 / B05[valid_mask]) /
        (1.0 / B03[valid_mask] + 1.0 / B05[valid_mask])
    )

    return ARI


def save_raster(array, ref_raster_path, output_path, nodata=0, dtype=gdal.GDT_Byte):
    """
    Sauvegarde un raster mono-bande.
    """
    ds_ref = gdal.Open(ref_raster_path)
    if ds_ref is None:
        raise IOError("Impossible d'ouvrir le raster de référence.")

    rows, cols = array.shape
    driver = gdal.GetDriverByName("GTiff")

    ds_out = driver.Create(output_path, cols, rows, 1, dtype)
    ds_out.SetGeoTransform(ds_ref.GetGeoTransform())
    ds_out.SetProjection(ds_ref.GetProjection())

    band = ds_out.GetRasterBand(1)
    band.WriteArray(array)
    band.SetNoDataValue(nodata)

    ds_out = None
    ds_ref = None
    return output_path


def save_multiband_raster(array_3d, ref_raster_path, output_path,
                          nodata=-9999, dtype=gdal.GDT_Float32):
    """
    Sauvegarde un raster multi-bandes.
    """
    ds_ref = gdal.Open(ref_raster_path)
    if ds_ref is None:
        raise IOError("Impossible d'ouvrir le raster de référence.")

    rows, cols, nb_bands = array_3d.shape
    driver = gdal.GetDriverByName("GTiff")

    ds_out = driver.Create(output_path, cols, rows, nb_bands, dtype)
    ds_out.SetGeoTransform(ds_ref.GetGeoTransform())
    ds_out.SetProjection(ds_ref.GetProjection())

    for i in range(nb_bands):
        band = ds_out.GetRasterBand(i + 1)
        band.WriteArray(array_3d[:, :, i])
        band.SetNoDataValue(nodata)

    ds_out = None
    ds_ref = None
    return output_path


def prepare_training_data(X_image, y_raster):
    """
    Prépare X et y pour scikit-learn à partir d'images raster.
    """
    rows, cols, nb_features = X_image.shape

    mask = y_raster > 0

    X_flat = X_image.reshape(rows * cols, nb_features).astype(np.float32)
    y_flat = y_raster.flatten().astype(np.int32)

    X = X_flat[mask.flatten(), :]
    y = y_flat[mask.flatten()]

    return X, y, mask
