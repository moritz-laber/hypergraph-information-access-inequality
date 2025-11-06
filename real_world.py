"""
Real World

Cleans real-world data, gender-labels data, and generates hypergraphs.
Also generates summary information about the hypergraphs.

Citations:
  Hospital
   - Philippe Vanhems, Alain Barrat, Ciro Cattuto, Jean-François
     Pinton, Nagham Khanafer, Corinne Régis, Byeul-a Kim, Brigitte
     Comte, and Nicolas Voirin. Estimating potential infection
     transmission routes in hospital wards using wearable proximity
     sensors. PloS one, 8(9):e73970, 2013.
  High School
   - Austin R. Benson, Rediet Abebe, Michael T. Schaub, Ali Jadbabaie,
     and Jon Kleinberg. Simplicial closure and higher-order link
     prediction. Proceedings of the National Academy of Sciences, 2018.
   - Rossana Mastrandrea, Julie Fournet, and Alain Barrat. Contact
     patterns in a high school: A comparison between data collected
     using wearable sensors, contact diaries and friendship surveys.
     PLOS ONE, 10(9):e0136497, 2015.
  Primary School
   - Philip S Chodrow, Nate Veldt, and Austin R Benson. Hypergraph
     clustering: from blockmodels to modularity. Science Advances, 2021.
   - Juliette Stehlé, Nicolas Voirin, Alain Barrat, Ciro Cattuto, Lorenzo
     Isella, Jean-François Pinton, Marco Quaggiotto, Wouter Van den Broeck,
     Corinne Régis, Bruno Lina, and Philippe Vanhems. High-resolutio
     measurements of face-to-face contact patterns in a primary school.
     PLoS ONE, 6(8):e23176, 2011.
  Senate Bill
   - Philip S Chodrow, Nate Veldt, and Austin R Benson. Hypergraph
     clustering: from blockmodels to modularity. Science Advances, 2021.
   - James H. Fowler. Connecting the congress: A study of cosponsorship
     networks. Political Analysis, 14(04):456–487, 2006.
   - James H. Fowler. Legislative cosponsorship networks in the US house
     and senate. Social Networks, 28(4):454–465, oct 2006.
  House Bills
   - Philip S Chodrow, Nate Veldt, and Austin R Benson. Hypergraph
     clustering: from blockmodels to modularity. Science Advances, 2021.
   - James H. Fowler. Connecting the congress: A study of cosponsorship
     networks. Political Analysis, 14(04):456–487, 2006.
   - James H. Fowler. Legislative cosponsorship networks in the US house
     and senate. Social Networks, 28(4):454–465, oct 2006.
  DBLP
   - DBLP. DBLP computer science bibliography.
     https://dblp.org/xml/release/dblp-2024-10-11.xml.gz, 2024.


SD - 2025/05/17
"""

import numpy as np
import pandas as pd
import igraph as ig
import pickle as pkl
import networkx as nx
from tqdm import tqdm
tqdm.pandas()
import statistics
import warnings
#import graph_tool as gt
#from graph_tool.topology import label_largest_component
from pathlib import Path
from typing import Optional
import requests
import shutil

from HyperGraph import *


def generate_benson_metadata(dataset: str, base_path: str='data/real_world/'):
    """
    Generate and save metadata files for the Benson house and senate bill datasets.

    Parameters:
        dataset: str, the name of the dataset
        base_path: str, file path to raw data

    Returns: none
    """
    # House bills
    if dataset == 'housebills':
        house_path = Path(base_path) / 'housebills'
        house_names = house_path / 'node-names-house-bills.txt'
        house_labels = house_path / 'node-labels-house-bills.txt'

        if not house_names.exists() or not house_labels.exists():
            raise FileNotFoundError("Missing house bill data: expected 'node-names-house-bills.txt' and 'node-labels-house-bills.txt'.")

        house_metadata = pd.read_csv(house_names, delimiter=';', header=None, names=['name'])
        house_party = pd.read_csv(house_labels, delimiter=';', header=None, names=['party'])

        house_metadata['id'] = list(range(1, len(house_metadata) + 1))
        house_metadata['party'] = house_party['party']
        house_metadata['party'] = house_metadata['party'].replace({'1': 'dem', '2': 'rep', 1: 'dem', 2: 'rep'})
        house_metadata.to_csv(house_path / 'metadata.csv', index=False)

    # Senate bills
    elif dataset == 'senatebills':
        senate_path = Path(base_path) / 'senatebills'
        senate_names = senate_path / 'node-names-senate-bills.txt'
        senate_labels = senate_path / 'node-labels-senate-bills.txt'

        if not senate_names.exists() or not senate_labels.exists():
            raise FileNotFoundError("Missing senate bill data: expected 'node-names-senate-bills.txt' and 'node-labels-senate-bills.txt'.")

        senate_metadata = pd.read_csv(senate_names, delimiter=';', header=None, names=['name'])
        senate_party = pd.read_csv(senate_labels, delimiter=';', header=None, names=['party'])

        senate_metadata['id'] = list(range(1, len(senate_metadata) + 1))
        senate_metadata['party'] = senate_party['party']
        senate_metadata['party'] = senate_metadata['party'].replace({'1': 'dem', '2': 'rep', 1: 'dem', 2: 'rep'})
        senate_metadata.to_csv(senate_path / 'metadata.csv', index=False)


def format_dblp_data(base_path: str='data/real_world/'):
    """
    Format DBLP author and edge data into a hypergraph-compatible structure.

    Parameters:
        base_path: str, file path to raw data

    Returns: none
    """
    dblp_path = Path(base_path) / 'dblp'
    author_fp = dblp_path / 'author.csv'
    edge_fp = dblp_path / 'edge_list.csv'

    if not author_fp.exists() or not edge_fp.exists():
        raise FileNotFoundError("Missing DBLP data: expected 'author.csv' and 'edge_list.csv'.")

    metadata = pd.read_csv(author_fp)
    metadata['node_id'] = list(metadata.index)
    metadata.to_csv(dblp_path / 'metadata.csv', index=False)

    df = pd.read_csv(edge_fp)
    if 'paper_id' not in df.columns or 'author_id' not in df.columns:
        raise ValueError("DBLP edge_list.csv must contain 'paper_id' and 'author_id' columns.")

    hyperedges = df.groupby('paper_id')['author_id'].apply(list).tolist()
    edge_df = pd.DataFrame({'hyperedges': [','.join(map(str, e)) for e in hyperedges if len(e) > 1]})
    edge_df.to_csv(dblp_path / 'dblp.csv', index=False)


def rework_indices(metadata: pd.DataFrame, edge_list: pd.DataFrame, folder: str, benson_dblp: bool = False):
    """
    Rework node indices for consistency, generate updated metadata, and rewrite edge list.

    Parameters:
        metadata: pd.DataFrame, node metadata
        edge_list: pd.DataFrame, edge data
        folder: str, name of the folder to save outputs
        benson_dblp: bool, True if processing DBLP or Benson datasets

    Returns: none
    """
    # Assign a new 'node_id' to each row in the metadata DataFrame (fix range to [0, num_nodes-1])
    metadata['node_id'] = list(metadata.index)
    node_mapping = dict(zip(metadata['id'], metadata['node_id']))
    metadata.to_csv(f'data/real_world/{folder}/metadata.csv', index=False)

    if not benson_dblp:
        # Save original edges; map edges to new node ids
        edge_list = edge_list.rename(columns={'id1': 'orig_id1', 'id2': 'orig_id2'})
        edge_list['id1'] = edge_list['orig_id1'].map(node_mapping)
        edge_list['id2'] = edge_list['orig_id2'].map(node_mapping)
        edge_list.to_csv(f'data/real_world/{folder}/{folder}.csv', index=False)

    else:
        # Save original edges; map edges to new node ids
        edge_list = edge_list.rename(columns={'hyperedges': 'orig_hyperedges'})
        hyperedges = edge_list['orig_hyperedges'].tolist()
        mapped_hyperedges = [[node_mapping[item] for item in sublist] for sublist in hyperedges]
        edge_list['hyperedges'] = [','.join(map(str, sublist)) for sublist in mapped_hyperedges]
        edge_list.to_csv(f'data/real_world/{folder}/{folder}.csv', index=False)


def get_LCC(metadata: pd.DataFrame, edge_list: pd.DataFrame, folder: str, base_path='data/real_world/', benson_dblp: bool = False):
    """
    Identify and annotate Largest Connected Component (LCC) membership.

    Parameters:
        metadata: DataFrame, node metadata
        edge_list: DataFrame, edge or hyperedge list
        folder: str, name of dataset folder
        base_path: str, root directory of data
        benson_dblp: bool, True if input is DBLP or Benson format

    Returns: none
    """

    metadata['node_id'] = metadata['node_id'].astype(int)
    nodes = list(metadata['node_id'])
    edge_list = edge_list.dropna()

    if not benson_dblp:
        edge_list['id1'] = edge_list['id1'].astype(int)
        edge_list['id2'] = edge_list['id2'].astype(int)

        g = ig.Graph()
        g.add_vertices(max(nodes) + 1)
        g.vs['name'] = nodes
        g.add_edges(list(zip(edge_list['id1'], edge_list['id2'])))
        lcc_nodes = set(g.connected_components().giant().vs['name'])

    else:
        g = gt.Graph(directed=False)
        g.add_vertex(len(nodes))
        hyperedges = [[int(i) for i in row.split(',')] for row in edge_list['hyperedges']]
        hyperedges = [edge for edge in hyperedges if len(edge) > 1]
        edge_pairs = (pair for edge in hyperedges for pair in combinations(edge, 2))
        g.add_edge_list(edge_pairs)
        lcc_mask = label_largest_component(g)
        lcc_nodes = {i for i, included in enumerate(lcc_mask.a) if included}

    metadata['in_lcc'] = metadata['node_id'].isin(lcc_nodes)
    metadata.to_csv(Path(base_path) / folder / 'metadata.csv', index=False)


def clean(datasets: List[str], data_dir: str = 'data/real_world/'):
    """
    Clean and reindex raw data, then compute largest connected component (LCC) membership.

    Parameters:
        datasets: list of str, dataset names to process
        data_dir: str, base directory path to datasets

    Returns: none
    """

    # Generate metadata if needed
    if 'dblp' in datasets:
        try:
            format_dblp_data(base_path=data_dir)
        except Exception as e:
            print("[Error] Failed to format DBLP data:", e)

    # Reindex and extract LCC
    for dataset in datasets:
        path = Path(data_dir) / dataset
        try:
            if dataset == 'primaryschool' or 'highschool' in dataset:
                metadata = pd.read_csv(path / 'metadata.csv', sep=r"\s+", header=None)
                metadata.columns = ["id", "class", "gender"]
                metadata = metadata[metadata['gender'] != 'Unknown']

                data = pd.read_csv(path / f'{dataset}.csv', sep=r"\s+", header=None)
                data.columns = ['time', 'id1', 'id2', 'class1', 'class2']
                data = data[(data['class1'] != 'Teachers') & (data['class2'] != 'Teachers')]

            #elif dataset == 'hospital':
            elif 'hospital' in dataset:
                data = pd.read_csv(path / f'{dataset}.csv', sep=r"\s+", header=None)
                data.columns = ['time', 'id1', 'id2', 'group1', 'group2']

                df1 = data[['id1', 'group1']].rename(columns={'id1': 'id', 'group1': 'group'})
                df2 = data[['id2', 'group2']].rename(columns={'id2': 'id', 'group2': 'group'})

                metadata = pd.concat([df1, df2], ignore_index=True).drop_duplicates()
                metadata = metadata.groupby('id')['group'].agg(lambda x: x.mode()[0]).reset_index()

            elif dataset == 'housebills' or dataset == 'senatebills':
                generate_benson_metadata(dataset, base_path=data_dir)

                metadata = pd.read_csv(path / 'metadata.csv')
                with open(path / f'{dataset}.csv', 'r') as f:
                    lines = f.read().splitlines()
                hyperedges = [list(map(int, line.split(','))) for line in lines]
                data = pd.DataFrame({'hyperedges': hyperedges})


            else:
                metadata = pd.read_csv(path / 'metadata.csv')
                data = pd.read_csv(path / f'{dataset}.csv')

        except FileNotFoundError as e:
            print(f"[Warning] Skipping '{dataset}' due to missing file:", e)
            continue

        benson = 'bill' in dataset
        dblp = dataset == 'dblp'
        benson_dblp = benson or dblp

        try:
            rework_indices(metadata, data, dataset, benson_dblp=benson_dblp)
            edge_list = pd.read_csv(path / f'{dataset}.csv')
            get_LCC(metadata, edge_list, dataset, base_path=data_dir, benson_dblp=benson_dblp)
        except Exception as e:
            print(f"[Error] Failed to process '{dataset}':", e)


def split_metadata_into_chunks(dataset: str, base_path: str, chunk_size: int=1000) -> int:
    """
    Splits metadata into chunks for parallel API calls.

    Parameters:
        dataset: str, name of dataset
        base_path: str, base path to data
        chunk_size: int, number of rows per chunk

    Returns:
        int: number of chunks created
    """
    metadata_fp = Path(base_path) / dataset / 'metadata.csv'
    if not metadata_fp.exists():
        raise FileNotFoundError(f"[split_metadata_into_chunks] Missing metadata.csv for {dataset}")

    df = pd.read_csv(metadata_fp)
    chunks = [df[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]

    for i, chunk in enumerate(chunks, 1):
        chunk.to_csv(Path(base_path) / dataset / f'to_gender_label_{i}.csv', index=False)

    print(f'Split {dataset} into {len(chunks)} chunks.')
    return len(chunks)


def label_gender_single(dataset: str, gender_api_key: str, genderize_io_key: str, base_path: str,
                        chunk_id: Optional[int] = None):
    """
    Gender-label a single dataset or chunk.

    Parameters:
        dataset: str, name of dataset (e.g., 'dblp')
        gender_api_key: str, GenderAPI key
        genderize_io_key: str, Genderize.io key
        base_path: str, base directory
        chunk_id: int, processes chunked metadata

    Returns: none
    """
    if chunk_id:
        fp = Path(base_path) / dataset / f'to_gender_label_{chunk_id}.csv'
    else:
        fp = Path(base_path) / dataset / 'metadata.csv'

    if not fp.exists():
        raise FileNotFoundError(f"[label_gender_single] Missing file: {fp}")

    df = pd.read_csv(fp)
    #df = df[['name', 'id', 'party', 'node_id', 'in_lcc', 'lcc_id']]
    #df = df.to_csv(fp, index=False)

    # Determine name column
    if 'first_name' in df.columns:
        col = 'first_name'
    elif 'new_first_name' in df.columns:
        col = 'new_first_name'
    elif 'name' in df.columns:
        df[['first_name', 'last_name']] = df['name'].apply(lambda x: pd.Series(split_author(x)))
        col = 'first_name'
    else:
        raise ValueError(f"No valid name field found in metadata for {dataset}")

    if not gender_api_key and not genderize_io_key:
        raise ValueError("At least one gender API key must be provided.")

    gender_api_dict, genderize_io_dict, df = get_gender(df, col, gender_api_key, genderize_io_key)

    if chunk_id:
        df.to_csv(fp, index=False)
        with open(Path(base_path) / dataset / f'gender_api_{chunk_id}.pkl', 'wb') as f:
            pkl.dump(gender_api_dict, f)
        with open(Path(base_path) / dataset / f'genderize_io_{chunk_id}.pkl', 'wb') as f:
            pkl.dump(genderize_io_dict, f)
    else:
        df.to_csv(Path(base_path) / dataset / 'metadata.csv', index=False)
        with open(Path(base_path) / dataset / 'gender_api.pkl', 'wb') as f:
            pkl.dump(gender_api_dict, f)
        with open(Path(base_path) / dataset / 'genderize_io.pkl', 'wb') as f:
            pkl.dump(genderize_io_dict, f)


def split_author(name: str):
    """
    Split a full name into first and last name for gender labeling.

    Parameters:
        name: str, Full name to split.

    Returns:
        first_name: Optional[str], First name extracted from the input string. May be None if not found.
        last_name: str, Remaining part of the name (may be empty).
    """

    # split the name by spaces
    parts = name.split()

    # check if the first non-space word is a single letter
    first_name = ""
    if len(parts) == 0:
        return "", ""

    # identify first name that isn't an initial
    i = 0
    while i < len(parts) and len(parts[i]) == 1:
        i += 1

    # return first and last name
    if i < len(parts):
        first_name = parts[i]
        last_name = ' '.join(parts[i + 1:])
    else:
        first_name = None
        last_name = ' '.join(parts)

    return first_name, last_name


def get_gender(df: pd.DataFrame, col: str, gender_api_key: str, genderize_io_key: str):
    """
    Loop through data to make calls to the gender-labeling APIs and format results.

    Parameters:
        df: dataframe, dataframe containing names to gender label
        col: string, name to column containing names to gender label
        gender_api_key: string, API key to access GenderAPI
        genderize_io_key: string, API key to access Genderize.io

    Returns:
        gender_api_dict: dictionary, results from GenderAPI
        genderize_io_dict: dictionary, results from Genderize.io
        df: dataframe, names to gender label with new gender columns
        """
    gender_api_dict = {}
    genderize_io_dict = {}

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        name = row[col]

        if gender_api_key == 'skip' and genderize_io_key == 'skip':
            # Assign dummy values
            df.at[idx, 'genderapi_gender'] = 'unknown'
            df.at[idx, 'genderapi_p'] = 0.0
            df.at[idx, 'genderapi_n'] = 0
            df.at[idx, 'genderizeio_gender'] = 'unknown'
            df.at[idx, 'genderizeio_p'] = 0.0
            df.at[idx, 'genderizeio_n'] = 0
            continue

        try:
            result = gender_api(name, gender_api_key)
            if result:
                df.at[idx, 'genderapi_gender'] = result.get('gender')
                df.at[idx, 'genderapi_p'] = result.get('accuracy') / 100
                df.at[idx, 'genderapi_n'] = result.get('samples')
                gender_api_dict[idx] = result
        except:
            pass

        try:
            result = genderize_io(name, genderize_io_key)
            if result:
                gender = result.get('gender') or 'unknown'
                df.at[idx, 'genderizeio_gender'] = gender
                df.at[idx, 'genderizeio_p'] = result.get('probability')
                df.at[idx, 'genderizeio_n'] = result.get('count')
                genderize_io_dict[idx] = result
        except:
            pass

    return gender_api_dict, genderize_io_dict, df


def gender_api(name, api_key):
    """
    Make calls to Genderize.io to gender-label names.

    Parameters:
        name: string, name to genderlabel
        api_key: string, API key to access Genderize.io

    Return:
        result: dictionary of results from API call
    """

    # set up parameters for API call
    url = f"https://gender-api.com/get"
    params = {
        'name': name,
        'key': api_key
    }

    # make API call and return the result
    response = requests.get(url, params=params)
    if response.status_code == 200:
        result = response.json()
        return result
    else:
        print(f"Failed to fetch data for {name}")
        return pd.Series([None, None])


def genderize_io(name: str, api_key: str):
    """
    Make calls to Genderize.io to gender-label names.

    Parameters:
        name: string, name to genderlabel
        api_key: string, API key to access Genderize.io

    Return:
        result: dictionary of results from API call
    """

    # set up parameters for API call
    url = f"https://api.genderize.io/"
    params = {
        'name': name,
        'apikey': api_key
    }

    # make API call and return the result
    response = requests.get(url, params=params)
    if response.status_code == 200:
        result = response.json()
        return result
    else:
        print(f"Failed to fetch data for {name}")
        return pd.Series([None, None])


def combine_chunked_gender_data(dataset: str, base_path: str, num_chunks: int):
    """
    Combines chunked gender-labeled files back into full metadata.

    Parameters:
        dataset: str, dataset name
        base_path: str, root directory
        num_chunks: int, number of chunks to merge

    Returns: none
    """
    dfs = []
    for i in range(1, num_chunks + 1):
        chunk_fp = Path(base_path) / dataset / f'to_gender_label_{i}.csv'
        if chunk_fp.exists():
            df = pd.read_csv(chunk_fp)
            dfs.append(df)
        else:
            print(f'[Warning] Missing chunk {i} for {dataset}.')

    full_df = pd.concat(dfs, ignore_index=True)
    full_df.to_csv(Path(base_path) / dataset / 'metadata.csv', index=False)
    print(f'Combined {len(dfs)} chunks for {dataset}.')


def label_gender_batch(datasets: List[str], gender_api_key: str=None, genderize_io_key: str=None,
                       base_path: str='data/real_world/', chunk_size: int=1000, split_chunks: bool=False):
    """
    Gender label datasets using GenderAPI and Genderize.io.

    Parameters:
        datasets: list of dataset names (e.g., ['dblp', 'housebills'])
        gender_api_key: str, API key for GenderAPI
        genderize_io_key: str, API key for Genderize.io
        base_path: str, base path to data directory
        chunk_size: int, number of rows per chunk if chunking
        split_chunks: bool, whether to chunk large datasets

    Returns: none
    """
    for dataset in datasets:
        print(f'\nProcessing: {dataset}')

        try:
            if dataset in ['dblp', 'aps']:
                if split_chunks:
                    num_chunks = split_metadata_into_chunks(dataset, base_path, chunk_size)
                else:
                    raise ValueError("split_chunks must be True for large datasets like dblp/aps.")

                for i in range(1, num_chunks + 1):
                    label_gender_single(
                        dataset,
                        gender_api_key,
                        genderize_io_key,
                        base_path,
                        chunk_id=i
                    )
                combine_chunked_gender_data(dataset, base_path, num_chunks)

            elif dataset in ['housebills', 'senatebills']:
                label_gender_single(dataset, gender_api_key, genderize_io_key, base_path)

                # Copy labeled file to new folder (e.g., senatebillsgender/)
                gendered_folder = dataset + 'gender'
                gendered_path = Path(base_path) / gendered_folder
                gendered_path.mkdir(parents=True, exist_ok=True)

                src = Path(base_path) / dataset / 'metadata.csv'
                dest = gendered_path / 'metadata.csv'
                src.rename(dest)

        except Exception as e:
            print(f"[Warning] Skipping {dataset} due to error: {e}")


def merge_gender_metadata(df: pd.DataFrame, name: str, base_path: str, merge_key: str='first_name'):
    """
    Merge gender predictions into original metadata.

    Parameters:
        df: pd.DataFrame, gender-labeled names
        name: str,  dataset name (e.g., 'dblp', 'housebillsgender')
        base_path: str,  base file path
        merge_key: str,  column in original metadata to merge on (e.g., 'new_first_name')

    Returns: none
    """
    metadata_fp = Path(base_path) / name / 'metadata.csv'
    metadata = pd.read_csv(metadata_fp)

    df['df_first_name'] = df['first_name']

    metadata = metadata.merge(
        df[['df_first_name', 'genderapi_gender', 'genderapi_p', 'genderapi_n', 'genderapi_p_female',
            'genderizeio_gender', 'genderizeio_p', 'genderizeio_n', 'genderizeio_p_female']],
        how='left',
        left_on=merge_key,
        right_on='df_first_name'
    )

    metadata.drop(columns=['df_first_name'], inplace=True)
    metadata.to_csv(metadata_fp, index=False)


def clean_gender_labels(df: pd.DataFrame, name: str, base_path: str='data/real_world/'):
    """
    Add gender probability columns and update metadata files.

    Parameters:
        df: pd.DataFrame, gender-labeled names
        name: str,  dataset name
        base_path: str,  base file path

    Returns: none
    """
    df['genderapi_p_female'] = np.where(
        df['genderapi_gender'] == 'female', df['genderapi_p'],
        np.where(df['genderapi_gender'] == 'male', 1 - df['genderapi_p'], 0.5)
    )

    df['genderizeio_p_female'] = np.where(
        df['genderizeio_gender'] == 'female', df['genderizeio_p'],
        np.where(df['genderizeio_gender'] == 'male', 1 - df['genderizeio_p'], 0.5)
    )

    # Determine appropriate merge key
    if name in ['dblp', 'aps']:
        merge_key = 'new_first_name'
    else:
        merge_key = 'first_name'

    merge_gender_metadata(df, name, base_path, merge_key)


def gender_label_and_clean_batch(datasets: List[str], gender_api_key: str, genderize_io_key: str,
                                 base_path: str='data/real_world/', chunk_size: int=1000, split_chunks: bool=False):
    """
    Label gender and clean metadata for multiple datasets.

    Parameters:
        datasets: list, dataset names (e.g., ['dblp', 'housebills'])
        gender_api_key: str, API key for GenderAPI
        genderize_io_key: str, API key for Genderize.io
        base_path: str, root path to data
        chunk_size: int, rows per chunk if splitting large datasets
        split_chunks: bool, whether to chunk large datasets like DBLP/APS

    Returns: none
    """
    for dataset in datasets:
        print(f'\nProcessing gender labeling for: {dataset}')

        try:
            # Step 1: Run labeling
            if dataset in ['dblp', 'aps']:
                if not split_chunks:
                    raise ValueError("split_chunks must be True for large datasets like dblp/aps.")
                num_chunks = split_metadata_into_chunks(dataset, base_path, chunk_size)

                for i in range(1, num_chunks + 1):
                    label_gender_single(
                        dataset,
                        gender_api_key,
                        genderize_io_key,
                        base_path,
                        chunk_id=i
                    )

                combine_chunked_gender_data(dataset, base_path, num_chunks)

            elif dataset in ['housebills', 'senatebills']:
                label_gender_single(dataset, gender_api_key, genderize_io_key, base_path)

                gendered_folder = dataset + 'gender'
                gendered_path = Path(base_path) / gendered_folder
                gendered_path.mkdir(parents=True, exist_ok=True)

                src = Path(base_path) / dataset / 'metadata.csv'
                dest = gendered_path / 'metadata.csv'
                shutil.copy(src, dest)

                pkl1 = Path(base_path) / dataset / 'gender_api.pkl'
                pkl2 = Path(base_path) / dataset / 'genderize_io.pkl'
                if pkl1.exists():
                    shutil.move(pkl1, gendered_path / 'gender_api.pkl')
                if pkl2.exists():
                    shutil.move(pkl2, gendered_path / 'genderize_io.pkl')

                original_csv = Path(base_path) / dataset / f"{dataset}.csv"
                renamed_csv = gendered_path / f"{gendered_folder}.csv"
                if original_csv.exists():
                    shutil.copy(original_csv, renamed_csv)

                # Update dataset name for Step 2
                dataset = gendered_folder

            # Step 2: Clean gender labels
            df = pd.read_csv(Path(base_path) / dataset / 'metadata.csv')
            clean_gender_labels(df, dataset, base_path)

        except Exception as e:
            print(f"[Warning] Skipping {dataset} due to error: {e}")


def generate_hypergraph(data: pd.DataFrame, metadata: pd.DataFrame, name: str, base_path: str, lcc: bool=True,
                        benson: bool=False, dblp: bool=False):
    """
    Generate hypergraphs from metadata.

    Parameters:
        data: pd.DataFrame, dataframe with data to make hypergraph
        metadata: pd.DataFrame, dataframe with data about nodes
        name: str, name of dataset
        base_path: str, base filepath for data
        lcc: bool, whether to use the LCC of the hypergraph
        benson: bool, whether the data is one of the Benson datasets
        dblp: bool, whether the data is the DBLP dataset

    Returns: none
    """
    if not benson and not dblp:
        hyperedges = []
        for _, group in data.groupby('time'):
            G = nx.Graph()
            G.add_edges_from(zip(group['id1'], group['id2']))
            hyperedges.extend(nx.find_cliques(G))
    else:
        data['hyperedges'] = data['hyperedges'].apply(lambda x: list(map(int, x.split(','))))
        hyperedges = data['hyperedges'].tolist()

    unique_edges = set(tuple(sorted(edge)) for edge in hyperedges)
    hyperedges = [list(edge) for edge in unique_edges]

    nodes, group, hyperedges = get_node_info(metadata, hyperedges, name, lcc, benson, dblp)

    H = HyperGraph(nodes=nodes, edges=hyperedges, group=group)
    print(f"{name}: {H.N} nodes, {H.M} edges, rank {H.rank}, group sizes {H.n}")

    out_path = Path(base_path) / name / 'hypergraphs'
    out_path.mkdir(parents=True, exist_ok=True)
    fp = out_path / f"{name}{'_lcc' if lcc else ''}_hg.pkl"
    with open(fp, 'wb') as f:
        pkl.dump(H, f)


def generate_pred_gender_hypergraph(data: pd.DataFrame, metadata: pd.DataFrame, name: str, base_path: str,
                                    lcc: bool=True, num_hypergraphs: int=1):
    """
    Generate hypergraphs from metadata.

    Parameters:
        data: pd.DataFrame, dataframe with data to make hypergraph
        metadata: pd.DataFrame, dataframe with data about nodes
        name: str, name of dataset
        base_path: str, base filepath for data
        lcc: bool, whether to use the LCC of the hypergraph
        num_hypergraphs: int, number of hypergraphs to generate

    Returns: none
    """
    data['hyperedges'] = data['hyperedges'].apply(lambda x: list(map(int, x.split(','))))
    hyperedges = data['hyperedges'].tolist()

    unique_edges = set(tuple(sorted(e)) for e in hyperedges)
    hyperedges = [list(t) for t in unique_edges]

    # Pass dblp=False to avoid NotImplementedError
    nodes, group_probs, hyperedges = get_node_info(metadata, hyperedges, name, lcc, benson=False, dblp=False, gender_probs=True)

    print(f'\nGenerating {num_hypergraphs} GenderAPI hypergraphs for {name}')
    out_path = Path(base_path) / name / 'hypergraphs_genderapi'
    out_path.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(num_hypergraphs)):
        group = np.random.binomial(1, group_probs['genderapi'])
        H = HyperGraph(nodes=nodes, edges=hyperedges, group=group)
        print(f'[{name} #{i}] Group sizes: {H.n}')
        fp = out_path / f"hg{'_lcc' if lcc else ''}_{i}.pkl"
        with open(fp, 'wb') as f:
            pkl.dump(H, f)

    print(f'\nGenerating {num_hypergraphs} Genderize.io hypergraphs for {name}')
    out_path = Path(base_path) / name / 'hypergraphs_genderizerio'
    out_path.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(num_hypergraphs)):
        group = np.random.binomial(1, group_probs['genderize_io'])
        H = HyperGraph(nodes=nodes, edges=hyperedges, group=group)
        print(f'[{name} #{i}] Group sizes: {H.n}')
        fp = out_path / f"hg{'_lcc' if lcc else ''}_{i}.pkl"
        with open(fp, 'wb') as f:
            pkl.dump(H, f)


def get_node_info(metadata: pd.DataFrame, hyperedges: List[List[int]], name: str, lcc: bool, benson: bool=False,
                  dblp: bool=False, gender_probs: bool=False):
    """
    Generate hypergraphs from metadata.

    Parameters:
        metadata: pd.DataFrame, dataframe with data about nodes
        name: str, name of dataset
        lcc: bool, whether to use the LCC of the hypergraph
        benson: bool, whether the data is one of the Benson datasets
        dblp: bool, whether the data is the DBLP dataset
        gender_probs: bool, whether the node labels are inferred gender labels

    Returns: none
    """
    if lcc:
        if 'in_lcc' not in metadata.columns:
            raise ValueError(f"'in_lcc' not found in metadata for {name}. Run the clean step first.")
        lcc_df = metadata[metadata['in_lcc'] == True].copy().reset_index(drop=True)

        if 'lcc_id' not in lcc_df.columns:
            print(f"[Info] Adding 'lcc_id' to metadata for {name}")
            lcc_df['lcc_id'] = range(len(lcc_df))
            metadata = metadata.merge(lcc_df[['node_id', 'lcc_id']], on='node_id', how='left')
            metadata.to_csv(f'data/real_world/{name}/metadata.csv', index=False)

        df = lcc_df
        nodes = list(df['lcc_id'].astype(int))
    else:
        df = metadata
        nodes = list(df['node_id'].astype(int))

    # Assign groups
    if gender_probs:
        group = {
            'genderapi': df['genderapi_p_female'],
            'genderize_io': df['genderizeio_p_female']
        }
    elif 'school' in name:
        group = [1 if g == 'F' else 0 for g in df['gender']]
    elif benson:
        group = [1 if p == 'rep' else 0 for p in df['party']]
    elif dblp:
        raise NotImplementedError("DBLP gender group not yet handled.")
    else:
        group = [1 if g == 'PAT' else 0 for g in df['group']]

    node_map = dict(zip(df['node_id'], df.get('lcc_id', df['node_id'])))
    reindexed_edges = [[node_map[n] for n in edge if n in node_map] for edge in hyperedges]

    return nodes, group, reindexed_edges


def generate_lcc_hypergraphs(datasets: List[str], base_path: str='data/real_world/', pred_gender: bool=False,
                             num_hypergraphs: int=1):
    """
    Generate hypergraph objects for each dataset.

    Parameters:
        datasets: list, dataset names
        base_path: str, base directory for reading/writing files
        pred_gender: bool, whether to use probabilistic gender labels
        num_hypergraphs: int, number of hypergraphs to generate if using predicted gender

    Returns: None
    """
    for dataset in datasets:
        benson = 'bill' in dataset
        dblp = dataset in ['dblp', 'aps']

        metadata_fp = Path(base_path) / dataset / 'metadata.csv'
        edge_fp = Path(base_path) / dataset / f'{dataset}.csv'

        if not metadata_fp.exists() or not edge_fp.exists():
            print(f'[Warning] Skipping {dataset} due to missing file.')
            continue

        metadata = pd.read_csv(metadata_fp)
        data = pd.read_csv(edge_fp)

        if pred_gender:
            generate_pred_gender_hypergraph(data, metadata, dataset, base_path, lcc=True, num_hypergraphs=num_hypergraphs)
        else:
            generate_hypergraph(data, metadata, dataset, base_path, lcc=True, benson=benson, dblp=dblp)


def compute_statistics(datasets: List[str], base_path: str='data/real_world/', output_path: str='data/real_world/real_hg_info.csv'):
    """
    Computes and saves summary statistics for standard hypergraphs (one per dataset).

    Parameters:
        datasets: list, dataset names
        base_path: str, base directory for input/output
        output_path: str, path to save summary CSV

    Returns: None
    """
    records = []

    for dataset in datasets:
        fp = Path(base_path) / dataset / 'hypergraphs' / f'{dataset}_lcc_hg.pkl'
        if not fp.exists():
            print(f"[Warning] Skipping {dataset} (missing hypergraph file).")
            continue

        with open(fp, 'rb') as f:
            H = pkl.load(f)

        degrees = H.degree
        group = H.group

        k_M = sum(degrees[n] for n in H.nodes if group[n] == 0)
        k_m = sum(degrees[n] for n in H.nodes if group[n] == 1)
        k = k_M + k_m

        row = {
            'dataset': dataset,
            'num_nodes': H.N,
            'p_m': H.n[1] / H.N,
            'edge_count': H.M,
            'k_bar': k / H.N,
            'k_bar_M': k_M / H.n[0],
            'k_bar_m': k_m / H.n[1],
            'power_inequality': H.power_inequality(),
            'moment_glass_ceiling': H.moment_glass_ceiling()
        }
        records.append(row)

    pd.DataFrame(records).to_csv(output_path, index=False)


def compute_pred_gender_statistics(datasets: List[str], num_hypergraphs: int=1, base_path: str='data/real_world/',
                                   output_path: str='data/real_world/real_hg_pred_gender_info.csv'):
    """
    Computes and saves summary statistics for probabilistic gender-labeled hypergraphs.

    Parameters:
        datasets: list, dataset names (e.g., ['dblp', 'senatebillsgender'])
        num_hypergraphs: int, number of sampled hypergraphs per dataset
        base_path: str, base directory for hypergraph files
        output_path: str, path to save summary CSV

    Returns: None
    """
    stats = []

    for dataset in datasets:
        for label_type in ['genderapi', 'genderizerio']:
            p_m_list, k_bar_M_list, k_bar_m_list, pi_list, mgc_list = [], [], [], [], []

            for i in tqdm(range(num_hypergraphs), desc=f"{dataset}-{label_type}"):
                fp = Path(base_path) / dataset / f'hypergraphs_{label_type}' / f'hg_lcc_{i}.pkl'
                if not fp.exists():
                    print(f"[Warning] Missing file: {fp}")
                    continue

                with open(fp, 'rb') as f:
                    H = pkl.load(f)

                degrees = H.degree
                group = H.group
                k_M = sum(degrees[n] for n in H.nodes if group[n] == 0)
                k_m = sum(degrees[n] for n in H.nodes if group[n] == 1)
                k = k_M + k_m

                p_m_list.append(H.n[1] / H.N)
                k_bar_M_list.append(k_M / H.n[0])
                k_bar_m_list.append(k_m / H.n[1])
                pi_list.append(H.power_inequality())
                mgc_list.append(H.moment_glass_ceiling())

            if not p_m_list:
                print(f"[Warning] No valid hypergraphs found for {dataset} ({label_type})")
                continue

            stats.append({
                'dataset': dataset,
                'labeling_type': label_type,
                'num_nodes': H.N,
                'edge_count': H.M,
                'k_bar': k / H.N,
                'avg_p_m': statistics.mean(p_m_list),
                'std_p_m': statistics.stdev(p_m_list) if len(p_m_list) > 1 else 0,
                'avg_k_bar_M': statistics.mean(k_bar_M_list),
                'std_k_bar_M': statistics.stdev(k_bar_M_list) if len(k_bar_M_list) > 1 else 0,
                'avg_k_bar_m': statistics.mean(k_bar_m_list),
                'std_k_bar_m': statistics.stdev(k_bar_m_list) if len(k_bar_m_list) > 1 else 0,
                'avg_power_inequality': statistics.mean(pi_list),
                'std_power_inequality': statistics.stdev(pi_list) if len(pi_list) > 1 else 0,
                'avg_moment_glass_ceiling': statistics.mean(mgc_list),
                'std_moment_glass_ceiling': statistics.stdev(mgc_list) if len(mgc_list) > 1 else 0
            })

    pd.DataFrame(stats).to_csv(output_path, index=False)


def main():
    warnings.filterwarnings('ignore')

    # Step 1: Clean data and compute LCCs
    clean(
        datasets=['highschool', 'primaryschool', 'hospital', 'senatebills', 'housebills', 'dblp', 'aps'],
        data_dir='data/real_world/'
    )
    
    # Step 2: Gender label (skip or comment if already labeled)
    GENDER_API_KEY = None  # replace with your API key for GenderAPI
    GENDERIZE_IO_KEY = None  # replace with your API key for genderize.io
    gender_label_and_clean_batch(
        datasets=['dblp', 'aps', 'senatebills', 'housebills'],
        gender_api_key=GENDER_API_KEY,
        genderize_io_key=GENDERIZE_IO_KEY,
        split_chunks=True,  # must be True for 'dblp' and 'aps'
        chunk_size=1000
    )
    
    # Step 3: Generate hypergraphs
    generate_lcc_hypergraphs(
        datasets=['highschool', 'primaryschool', 'hospital', 'senatebills', 'housebills'],
        base_path='data/real_world/',
        pred_gender=False,
        num_hypergraphs=1
    )
    generate_lcc_hypergraphs(
        datasets=['senatebillsgender', 'housebillsgender', 'dblp', 'aps'],
        base_path='data/real_world/',
        pred_gender=True,
        num_hypergraphs=1000
    )
    
    # Step 4: Compute summary statistics
    compute_statistics(
        datasets=['highschool', 'hospital', 'primaryschool', 'housebills', 'senatebills']
    )
    compute_pred_gender_statistics(
        datasets=['dblp', 'aps', 'senatebillsgender', 'housebillsgender'],
        num_hypergraphs=1000
    )


if __name__=='__main__':
    main()