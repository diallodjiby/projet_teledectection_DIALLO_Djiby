# -*- coding: utf-8 -*-
"""
Fonctions personnalisées pour le projet de télédétection
Auteur : Djiby Diallo

Complément des fonctions fournies dans libsigma
"""

import numpy as np
import pandas as pd
import os
from osgeo import gdal, ogr
import matplotlib.pyplot as plt

# =========================================================
# ===================== ARI ===============================
# =========================================================

def calculate_ari(B03, B05, nodata=-9999):
    if B03.shape != B05.shape:
        raise ValueError("Dimensions incompatibles B03 / B05")

    ARI = np.full(B03.shape, nodata, dtype=np.float32)
    valid_mask = (B03 != nodata) & (B05 != nodata) & (B03 > 0) & (B05 > 0)

    with np.errstate(divide='ignore', invalid='ignore'):
        num = (1.0 / B03[valid_mask] - 1.0 / B05[valid_mask])
        den = (1.0 / B03[valid_mask] + 1.0 / B05[valid_mask])
        ARI[valid_mask] = np.where(den != 0, num / den, nodata)

    return ARI


def extract_ari_statistics_by_class(ari_series, strates_raster, classes, nodata=-9999):
    if ari_series.shape[:2] != strates_raster.shape:
        raise ValueError("Dimensions incompatibles ARI / strates")

    n_classes = len(classes)
    n_dates = ari_series.shape[2]

    moyennes = np.full((n_classes, n_dates), np.nan, dtype=np.float32)
    ecarts_types = np.full((n_classes, n_dates), np.nan, dtype=np.float32)

    for i, classe in enumerate(classes):
        mask = strates_raster == classe

        for d in range(n_dates):
            vals = ari_series[:, :, d][mask]
            vals = vals[vals != nodata]

            if vals.size > 0:
                moyennes[i, d] = np.mean(vals)
                ecarts_types[i, d] = np.std(vals)

    return moyennes, ecarts_types


def create_ari_phenology_plot(moyennes, ecarts_types, dates, class_info, output_path):
    plt.figure(figsize=(12, 7))

    for i, (classe, (name, col)) in enumerate(class_info.items()):
        plt.plot(dates, moyennes[i], label=name, color=col, marker='o', linewidth=2)
        plt.fill_between(
            dates,
            moyennes[i] - ecarts_types[i],
            moyennes[i] + ecarts_types[i],
            alpha=0.15,
            color=col
        )

    plt.xlabel("Date")
    plt.ylabel("ARI moyen")
    plt.title("Évolution temporelle de l'ARI par strate")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print("✅ Graphique ARI sauvegardé :", output_path)


# =========================================================
# ===================== FEATURES ==========================
# =========================================================

def create_feature_names(bandes, dates, prefix="t"):
    if prefix == "t":
        return [f"{b}_{prefix}{i+1}" for b in bandes for i in range(len(dates))]
    else:
        return [f"{b}_{d}" for b in bandes for d in dates]


def analyze_feature_importance(model, feature_names, top_n=20, save_path=None):
    if not hasattr(model, "feature_importances_"):
        raise AttributeError("Le modèle ne contient pas feature_importances_")

    df = pd.DataFrame({
        "Variable": feature_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    print(f"\nTop {top_n} variables :")
    print(df.head(top_n).to_string(index=False))

    if save_path:
        plt.figure(figsize=(10, 8))
        top = df.head(top_n)
        plt.barh(top["Variable"], top["Importance"], color='steelblue')
        plt.gca().invert_yaxis()
        plt.xlabel("Importance")
        plt.title("Variables les plus importantes")
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"✅ Graphique d'importance sauvegardé : {save_path}")

    return df


# =========================================================
# ===================== RASTER ============================
# =========================================================

def rasterize_vector(vector_path, ref_raster_path, output_path, attribute="strate"):
    ds_ref = gdal.Open(ref_raster_path)
    if ds_ref is None:
        raise IOError("Impossible d'ouvrir le raster de référence")

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

    vect_ds = ogr.Open(vector_path)
    layer = vect_ds.GetLayer()

    gdal.RasterizeLayer(ds_out, [1], layer, options=[f"ATTRIBUTE={attribute}"])

    ds_out = None
    vect_ds = None
    ds_ref = None

    print(f"✅ Rasterisation terminée : {output_path}")
    return output_path


def save_ari_series(ari_array, reference_raster_path, output_path, nodata=-9999):
    ds_ref = gdal.Open(reference_raster_path)
    rows, cols, nb_dates = ari_array.shape

    driver = gdal.GetDriverByName("GTiff")
    ds_out = driver.Create(output_path, cols, rows, nb_dates, gdal.GDT_Float32,
                           options=["COMPRESS=LZW", "PREDICTOR=2"])

    ds_out.SetGeoTransform(ds_ref.GetGeoTransform())
    ds_out.SetProjection(ds_ref.GetProjection())

    for i in range(nb_dates):
        band = ds_out.GetRasterBand(i + 1)
        band.WriteArray(ari_array[:, :, i])
        band.SetNoDataValue(nodata)
        band.SetDescription(f"ARI_t{i+1}")

    ds_out = None
    ds_ref = None

    print(f"✅ Série temporelle ARI sauvegardée : {output_path}")


def save_multiband_raster(array, reference_raster_path, output_path, nodata=-9999):
    ds_ref = gdal.Open(reference_raster_path)
    rows, cols, bands = array.shape

    driver = gdal.GetDriverByName("GTiff")
    ds_out = driver.Create(output_path, cols, rows, bands, gdal.GDT_Float32,
                           options=["COMPRESS=LZW", "PREDICTOR=2"])

    ds_out.SetGeoTransform(ds_ref.GetGeoTransform())
    ds_out.SetProjection(ds_ref.GetProjection())

    for b in range(bands):
        band = ds_out.GetRasterBand(b + 1)
        band.WriteArray(array[:, :, b])
        band.SetNoDataValue(nodata)

    ds_out = None
    ds_ref = None

    print(f"✅ Raster multibande sauvegardé : {output_path}")


def save_classification_map(predicted_array, reference_raster_path, output_path, nodata_value=0):
    ds_ref = gdal.Open(reference_raster_path)
    rows, cols = predicted_array.shape

    driver = gdal.GetDriverByName("GTiff")
    ds_out = driver.Create(output_path, cols, rows, 1, gdal.GDT_Byte,
                           options=["COMPRESS=LZW", "PREDICTOR=2"])

    ds_out.SetGeoTransform(ds_ref.GetGeoTransform())
    ds_out.SetProjection(ds_ref.GetProjection())

    band = ds_out.GetRasterBand(1)
    band.WriteArray(predicted_array.astype(np.uint8))
    band.SetNoDataValue(nodata_value)

    color_table = gdal.ColorTable()
    color_table.SetColorEntry(1, (255, 0, 0))
    color_table.SetColorEntry(2, (0, 255, 0))
    color_table.SetColorEntry(3, (255, 0, 255))
    color_table.SetColorEntry(4, (0, 128, 0))

    band.SetColorTable(color_table)
    band.SetDescription("Classification des strates")

    ds_out = None
    ds_ref = None

    print(f"✅ Carte classifiée sauvegardée : {output_path}")


# =========================================================
# ===================== ML ================================
# =========================================================

def prepare_training_data(X_image, y_raster, nodata_label=0):
    if X_image.shape[:2] != y_raster.shape:
        raise ValueError("Dimensions incompatibles image / labels")

    mask = y_raster != nodata_label

    X = X_image[mask]
    Y = y_raster[mask].reshape(-1, 1)

    return X, Y, mask


def report_from_dict_to_df(report_dict):
    df = pd.DataFrame(report_dict).T
    for col in df.columns:
        if col != 'support':
            df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def predict_by_blocks(model, image_array, block_size=1000):
    rows, cols, bands = image_array.shape
    classified = np.zeros((rows, cols), dtype=np.uint8)

    for y in range(0, rows, block_size):
        y_end = min(y + block_size, rows)
        for x in range(0, cols, block_size):
            x_end = min(x + block_size, cols)

            block = image_array[y:y_end, x:x_end, :]
            flat_block = block.reshape(-1, bands)

            valid_mask = ~np.any(flat_block == -9999, axis=1)
            predictions = np.zeros(flat_block.shape[0], dtype=np.uint8)

            if np.any(valid_mask):
                predictions[valid_mask] = model.predict(flat_block[valid_mask])

            classified[y:y_end, x:x_end] = predictions.reshape(block.shape[0], block.shape[1])

    return classified
