"""
Fonctions personnalisées pour le projet de télédétection
Auteur : Djiby Diallo
Cours : Télédétection Approfondissement - Qualité et fouille de données
"""

import numpy as np
import os
from osgeo import gdal, ogr


def create_stack(file_list, output_path):
    """
    Crée un stack multi-temporel à partir d'une liste de fichiers raster.
    
    Parameters
    ----------
    file_list : list
        Liste des chemins vers les fichiers raster à empiler
    output_path : str
        Chemin du fichier de sortie (format GeoTIFF)
    
    Returns
    -------
    str
        Chemin du fichier créé
    
    Example
    -------
    >>> files = ['B03_date1.tif', 'B03_date2.tif']
    >>> create_stack(files, 'stack_B03.tif')
    """
    vrt_temp = output_path.replace('.tif', '.vrt')
    
    # Créer un VRT (Virtual Raster)
    gdal.BuildVRT(vrt_temp, file_list, separate=True)
    
    # Convertir le VRT en GeoTIFF
    gdal.Translate(output_path, vrt_temp, format="GTiff")
    
    # Supprimer le fichier temporaire
    os.remove(vrt_temp)
    
    return output_path


def rasterize_vector(vector_path, ref_raster_path, output_path, attribute="strate"):
    """
    Rasterise un shapefile en utilisant un raster de référence pour la géométrie.
    
    Parameters
    ----------
    vector_path : str
        Chemin vers le fichier shapefile
    ref_raster_path : str
        Chemin vers le raster de référence (pour projection, résolution, emprise)
    output_path : str
        Chemin du raster de sortie
    attribute : str, optional
        Nom de l'attribut à rasteriser (par défaut: "strate")
    
    Returns
    -------
    str
        Chemin du fichier créé
    
    Example
    -------
    >>> rasterize_vector('echantillons.shp', 'ref.tif', 'strates.tif')
    """
    # Ouvrir le raster de référence
    ds_ref = gdal.Open(ref_raster_path)
    gt = ds_ref.GetGeoTransform()
    proj = ds_ref.GetProjection()
    cols = ds_ref.RasterXSize
    rows = ds_ref.RasterYSize
    
    # Créer le raster de sortie
    driver = gdal.GetDriverByName("GTiff")
    ds_out = driver.Create(output_path, cols, rows, 1, gdal.GDT_Byte)
    ds_out.SetGeoTransform(gt)
    ds_out.SetProjection(proj)
    
    # Initialiser la bande avec nodata=0
    band = ds_out.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.Fill(0)
    
    # Ouvrir le shapefile
    ds_vect = ogr.Open(vector_path)
    layer = ds_vect.GetLayer()
    
    # Rasteriser
    gdal.RasterizeLayer(
        ds_out, 
        [1], 
        layer, 
        options=[f"ATTRIBUTE={attribute}"]
    )
    
    # Fermer les datasets
    ds_out = None
    ds_ref = None
    ds_vect = None
    
    return output_path


def extract_stats_by_class(image, mask, classes, nodata=-9999):
    """
    Extrait les statistiques (moyenne et écart-type) par classe pour chaque date.
    
    Parameters
    ----------
    image : numpy.ndarray
        Image 3D de shape (rows, cols, nb_dates)
    mask : numpy.ndarray
        Masque 2D des classes de shape (rows, cols)
    classes : list
        Liste des identifiants de classes
    nodata : float, optional
        Valeur de nodata à exclure (par défaut: -9999)
    
    Returns
    -------
    tuple
        (moyennes, ecarts_types) : deux arrays de shape (nb_classes, nb_dates)
    
    Example
    -------
    >>> ARI = np.random.rand(100, 100, 6)
    >>> strates = np.random.randint(1, 5, (100, 100))
    >>> moy, std = extract_stats_by_class(ARI, strates, [1, 2, 3, 4])
    """
    nb_dates = image.shape[2]
    nb_classes = len(classes)
    
    # Initialiser les tableaux de résultats
    moyennes = np.zeros((nb_classes, nb_dates))
    ecarts_types = np.zeros((nb_classes, nb_dates))
    
    # Pour chaque classe
    for i, classe in enumerate(classes):
        # Créer le masque pour cette classe
        mask_classe = (mask == classe)
        
        # Pour chaque date
        for date in range(nb_dates):
            # Extraire les valeurs de cette classe à cette date
            valeurs = image[:, :, date][mask_classe]
            
            # Filtrer les valeurs nodata
            valeurs = valeurs[valeurs != nodata]
            
            # Calculer les statistiques si on a des valeurs
            if len(valeurs) > 0:
                moyennes[i, date] = np.mean(valeurs)
                ecarts_types[i, date] = np.std(valeurs)
            else:
                moyennes[i, date] = np.nan
                ecarts_types[i, date] = np.nan
    
    return moyennes, ecarts_types


def calculate_ari(B03, B05, nodata=-9999):
    """
    Calcule l'indice ARI (Anthocyanin Reflectance Index).
    
    Formule : ARI = (1/B03 - 1/B05) / (1/B03 + 1/B05)
    
    Parameters
    ----------
    B03 : numpy.ndarray
        Bande verte (B03) de Sentinel-2
    B05 : numpy.ndarray
        Bande red-edge (B05) de Sentinel-2
    nodata : float, optional
        Valeur à assigner aux pixels invalides (par défaut: -9999)
    
    Returns
    -------
    numpy.ndarray
        Indice ARI calculé, même shape que les entrées
    
    Example
    -------
    >>> B03 = np.random.rand(100, 100, 6) * 3000
    >>> B05 = np.random.rand(100, 100, 6) * 3000
    >>> ARI = calculate_ari(B03, B05)
    """
    # Désactiver les warnings pour division par zéro
    np.seterr(divide='ignore', invalid='ignore')
    
    # Calculer l'ARI
    ARI = (1.0/B03 - 1.0/B05) / (1.0/B03 + 1.0/B05)
    
    # Remplacer les NaN par nodata
    ARI[np.isnan(ARI)] = nodata
    
    return ARI


def save_raster(array, ref_raster_path, output_path, nodata=0, dtype=gdal.GDT_Byte):
    """
    Sauvegarde un array numpy en raster GeoTIFF en utilisant un raster de référence.
    
    Parameters
    ----------
    array : numpy.ndarray
        Array 2D à sauvegarder
    ref_raster_path : str
        Chemin vers le raster de référence (pour géoréférencement)
    output_path : str
        Chemin du fichier de sortie
    nodata : int/float, optional
        Valeur de nodata (par défaut: 0)
    dtype : gdal.DataType, optional
        Type de données GDAL (par défaut: Byte)
    
    Returns
    -------
    str
        Chemin du fichier créé
    
    Example
    -------
    >>> predictions = np.random.randint(1, 5, (100, 100))
    >>> save_raster(predictions, 'ref.tif', 'carte.tif')
    """
    # Ouvrir le raster de référence
    ds_ref = gdal.Open(ref_raster_path)
    
    # Créer le raster de sortie
    driver = gdal.GetDriverByName("GTiff")
    rows, cols = array.shape
    
    ds_out = driver.Create(output_path, cols, rows, 1, dtype)
    ds_out.SetGeoTransform(ds_ref.GetGeoTransform())
    ds_out.SetProjection(ds_ref.GetProjection())
    
    # Écrire les données
    band = ds_out.GetRasterBand(1)
    band.WriteArray(array)
    band.SetNoDataValue(nodata)
    
    # Fermer les datasets
    ds_out = None
    ds_ref = None
    
    return output_path


def save_multiband_raster(array_3d, ref_raster_path, output_path, 
                          nodata=-9999, dtype=gdal.GDT_Float32):
    """
    Sauvegarde un array 3D en raster multi-bandes.
    
    Parameters
    ----------
    array_3d : numpy.ndarray
        Array 3D de shape (rows, cols, nb_bands)
    ref_raster_path : str
        Chemin vers le raster de référence
    output_path : str
        Chemin du fichier de sortie
    nodata : float, optional
        Valeur de nodata (par défaut: -9999)
    dtype : gdal.DataType, optional
        Type de données GDAL (par défaut: Float32)
    
    Returns
    -------
    str
        Chemin du fichier créé
    
    Example
    -------
    >>> ARI = np.random.rand(100, 100, 6)
    >>> save_multiband_raster(ARI, 'ref.tif', 'ARI_serie.tif')
    """
    # Ouvrir le raster de référence
    ds_ref = gdal.Open(ref_raster_path)
    
    # Dimensions
    rows, cols, nb_bands = array_3d.shape
    
    # Créer le raster de sortie
    driver = gdal.GetDriverByName("GTiff")
    ds_out = driver.Create(output_path, cols, rows, nb_bands, dtype)
    ds_out.SetGeoTransform(ds_ref.GetGeoTransform())
    ds_out.SetProjection(ds_ref.GetProjection())
    
    # Écrire chaque bande
    for i in range(nb_bands):
        band = ds_out.GetRasterBand(i + 1)
        band.WriteArray(array_3d[:, :, i])
        band.SetNoDataValue(nodata)
    
    # Fermer les datasets
    ds_out = None
    ds_ref = None
    
    return output_path


def prepare_training_data(X_image, y_raster):
    """
    Prépare les données d'entraînement pour scikit-learn à partir d'images.
    
    Parameters
    ----------
    X_image : numpy.ndarray
        Image des features de shape (rows, cols, nb_features)
    y_raster : numpy.ndarray
        Raster des labels de shape (rows, cols)
    
    Returns
    -------
    tuple
        (X, y, mask) où :
        - X : array 2D (nb_samples, nb_features)
        - y : array 1D (nb_samples,)
        - mask : masque booléen des pixels étiquetés
    
    Example
    -------
    >>> X_img = np.random.rand(100, 100, 66)
    >>> y_img = np.random.randint(0, 5, (100, 100))
    >>> X, y, mask = prepare_training_data(X_img, y_img)
    """
    rows, cols, nb_features = X_image.shape
    
    # Créer un masque des pixels étiquetés (> 0)
    mask = y_raster > 0
    
    # Reshape en 2D
    X_flat = X_image.reshape(rows * cols, nb_features)
    y_flat = y_raster.flatten()
    
    # Extraire seulement les pixels avec label
    X = X_flat[mask.flatten(), :]
    y = y_flat[mask.flatten()]
    
    return X, y, mask