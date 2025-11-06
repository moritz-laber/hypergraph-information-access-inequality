"""
This script is used to preprocess the APS dataset.
"""

import json
import os
from typing import Dict, List, Tuple, Union

import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def read_one_aps_json(file: str) -> Tuple[Dict[str, Dict[str, Union[str, List[str], bool]]], str]:
    """
    Read in a single APS metadata file and return an author data dictionary and a 
    list of author names.
        - firstname: author name
        - surname: author name
        - institution: author institution
    - paper id

    :param file: The path to the APS metadata file.
    :return: A tuple containing an author data dictionary and a list of author names.

    """
    # read in the metadata
    with open(file, 'r') as f:
        data = json.load(f)
    authors = data.get("authors", [])
    affiliations = data.get("affiliations", [])

    res = {}
    for author_dict in authors:
        fullname = author_dict.get("name", "")
        # authors.append(fullname)
        firstname = author_dict.get("firstname", "")
        surname = author_dict.get("surname", "")
        affs = author_dict.get("affiliationIds", [])
        if len(affs) == 0 or firstname == "" or surname == "":
            return None, None
        institutions = [aff["name"] for aff in affiliations if aff["id"] in affs]
        # physics = sum(["physics" in aff.lower() for aff in institutions]) > 0
        res[fullname] = {
            "fullname": fullname,
            "firstname": firstname,
            "surname": surname,
            # "physics": physics, 
            "institutions": institutions
            }

    return res, data["id"]


def read_aps_json(journal: str, output:bool=True, include_large:bool=True) -> Dict[str, Dict[str, Union[str, List[str], bool]]]:
    """
    Reads the APS metadata and returns a dictionary of the data.
    Also dumps the data into a json in the journal folder.
    :param journal: The folder containing a journal from APS.
    :param output: If True, output the data to a json file.
    :param include_large: If True, include papers with more than 10 authors.

    :return: Dict. The dictionary should contain the following keys:
    - metadata: a list of dictionaries containing author metadata with:
        - id: author ID
        - firstname: author name
        - surname: author name
        - institution: author institution
    """
    results = {}
    # get the metadata
    # Append to metadata.json
    metadata_file = os.path.join(journal, 'metadata.json')
    dirs = os.listdir(journal)
    dirs = [os.path.join(journal, dir) for dir in dirs if dir != '.DS_Store']
    dirs = [dir for dir in dirs if os.path.isdir(dir)]
    
    # num_authors = []
    # counter_all = 0
    for dir in dirs:
        for file in os.listdir(dir):
            try:
                file_dict, id = read_one_aps_json(os.path.join(dir, file))
            except Exception as e:
                print(e)
                print(f"Error reading {os.path.join(dir, file)}")
                continue

            # Add to results
            if file_dict is not None:
                # counter_all += 1
                if len(file_dict) <= 10 or include_large:
                    results[id] = file_dict
                # else:
                    # num_authors.append(len(file_dict))

    # output the metadata
    if output:
        with open(metadata_file, 'w') as f:
            json.dump(results, f, indent=4)

    return results


def format_aps_data(data_path, delimiter= ';\t', verbose=False, output_journal=False):
    """
    Formats the APS author and edge data for use in the simulation.
    Writes edge data (aps.csv) and metadata (metadata.csv) to a files.
    :param data_path: The path to the overall APS directory.
    :param delimiter: The delimiter for the metadata file.
    :param verbose: If True, print out the journal being read.
    :param output_journal: If True, output the data to a json file journal by journal.

    :return: None
    """
    # get all the journals
    journals = os.listdir(data_path)
    journals = [os.path.join(data_path, journal) for journal in journals if journal != '.DS_Store' ]
    journals = [journal for journal in journals if os.path.isdir(journal)]

    # get all the data from different journals
    all_journals = []
    for journal in tqdm.tqdm(journals):
        if verbose:
            print(f"Reading {journal}")
        all_journals.append(read_aps_json(journal, output=output_journal))

    # write the data to csv file called metadata.csv
    metadata_file = 'data/real_world/aps/metadata_raw.csv'
    with open(metadata_file, 'w') as f:
        # write the header
        f.write(f"id{delimiter}author{delimiter}firstname{delimiter}surname{delimiter}institutions\n")
        for journal in all_journals:
            for id, data in journal.items():
                for author, author_data in data.items():
                    f.write(f"{id}{delimiter}{author}{delimiter}{author_data['firstname']}{delimiter}{author_data['surname']}{delimiter}{author_data['institutions']}\n")

    # return the metadata dataframe 
    return pd.read_csv('data/real_world/aps/metadata_raw.csv', sep=delimiter, engine='python')


def format_dataframe(
    df_orig: pd.DataFrame, 
    edge_list_file: str = 'data/real_world/aps/aps-dataset-citations-2022.csv',
    ) -> pd.DataFrame:
    """
    Clean author names according to https://arxiv.org/pdf/2407.11909 fig. S1 c.
    """
    print('cleaning author names')
    # make a copy of the dataframe
    df = df_orig.copy()
    # split the affiliations column into a list
    df['affiliations'] = df['affiliations'].apply(lambda x: x[2:-2])
    df['affiliations'] = df['affiliations'].apply(lambda x: x.split("\', \'"))
    df['n_affs'] = df['affiliations'].apply(lambda x: len(x))

    # clean first and last names
    df['first_name'] = df['first_name'].apply(lambda x: x.strip())
    df['surname'] = df['surname'].apply(lambda x: x.strip())
    df['first_name'] = df['first_name'].replace(' ', np.nan)
    df['first_name'] = df['first_name'].replace('', np.nan)
    df = df.dropna()
    # get the first initial of the first name
    df['first_initial'] = df['first_name'].apply(lambda x: x[0])

    # add in coauthors for each id, get the other authors first initial and surname
    coauthors = df.groupby('id')[['first_initial', 'surname']].agg(list).reset_index()

    # zip together the first initial and surname
    coauthors['names'] = coauthors.apply(lambda x: list(zip(x['first_initial'], x['surname'])), axis=1)
    # print(coauthors.head())
    coauthors = coauthors.drop(
        columns=['first_initial', 'surname']
        ).rename(
            columns=
            {
                'names': 'coauthors', 
            }
            )

    # merge coauthors with the original dataframe
    print('merging coauthors')
    df = df.merge(coauthors, on='id')
    # remove the author from the coauthors list
    df['coauthors'] = df.apply(lambda x: [i for i in x['coauthors'] if i != (x['first_initial'], x['surname'])], axis=1)


    # get citations
    print('getting citations')
    citations = pd.read_csv(edge_list_file)
    # group by citing doi and create a list of cited dois
    citations = citations.groupby('citing_doi')['cited_doi'].apply(list).reset_index().rename(columns={'citing_doi': 'id'})
    # merge into the dataframe
    df = df.merge(citations, on='id', how='left')
    df = df.rename(columns={'cited_doi': 'citations'})

    # save the checkpoint
    df['unique_author_id'] = df.index
    return df


###############################################################################
# Merge Authors
###############################################################################
def merge_authors(df2, verbose=False):
    # 0) normalize & coerce missing lists
    df = df2.copy()
    df['surname']       = df['surname'].str.lower().str.strip()
    df['first_initial'] = df['first_name'].str.lower().str.strip().str[:1].fillna('')
    df['block']         = df['surname'] + "_" + df['first_initial']
    for col in ('coauthors','affiliations','citations'):
        df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])

    # 1) re‐index to get node IDs 0…N-1
    if verbose:
        print('Resetting index...')
    df = df.reset_index(drop=True).rename_axis('node').reset_index()
    N = len(df)
    rows, cols = [], []

    # 2) shared coauthors → edges
    if verbose:
        print('Finding shared coauthors...')
    co = df[['node','block','coauthors']].explode('coauthors').dropna(subset=['coauthors'])
    cm = co.merge(co, on=['block','coauthors'])
    mask = cm['node_x'] < cm['node_y']
    rows.extend(cm.loc[mask, 'node_x'])
    cols.extend(cm.loc[mask, 'node_y'])

    # 3) shared affiliations → edges
    if verbose:
        print('Finding shared affiliations...')
    aff = df[['node','block','affiliations']].explode('affiliations').dropna(subset=['affiliations'])
    am = aff.merge(aff, on=['block','affiliations'])
    mask = am['node_x'] < am['node_y']
    rows.extend(am.loc[mask, 'node_x'])
    cols.extend(am.loc[mask, 'node_y'])

    # 4) citations → edges
    if verbose:
        print('Finding citations...')
    papers = (
        df[['node','block','id']]
        .rename(columns={'node':'author_node','id':'paper_id'})
    )
    cit = (
        df[['node','block','citations']]
          .explode('citations')
          .dropna(subset=['citations'])
          .rename(columns={'node':'citing_node','citations':'cited_paper_id'})
    )
    cm2 = cit.merge(
        papers,
        left_on=['block','cited_paper_id'],
        right_on=['block','paper_id'],
        suffixes=('','_cited')
    )
    # drop self-citations only
    cm2 = cm2[cm2['citing_node'] != cm2['author_node']]
    # add _all_ directed pairs; adj + adj.T will give you a clean undirected graph
    rows.extend(cm2['citing_node'])
    cols.extend(cm2['author_node'])


    # 5) build sparse adjacency & find components
    if verbose:
        print('Building sparse adjacency...')
    data = np.ones(len(rows), bool)
    adj  = coo_matrix((data, (rows, cols)), shape=(N,N))
    adj  = adj + adj.T
    if verbose:
        print('Finding connected components...')
    n_comp, labels = connected_components(adj, directed=False)

    # 6) aggregate per component
    if verbose:
        print('Aggregating per component...')
    df['component'] = labels
    grouped = df.groupby('component', sort=False)
    merged = pd.DataFrame({
        'members':         grouped['unique_author_id'].apply(lambda s: sorted(s)),
        'merged_author_id': grouped['unique_author_id'].min(),
        'all_first_names': grouped['first_name'].apply(lambda s: sorted(set(s))),
        'all_surnames':    grouped['surname'].apply(lambda s: sorted(set(s))),
        'paper_ids':       grouped['id'].apply(lambda s: sorted({x for sub in s for x in (sub if isinstance(sub,list) else [sub])})),
        'coauthors':       grouped['coauthors'].apply(lambda s: sorted({c for sub in s for c in sub})),
        'affiliations':    grouped['affiliations'].apply(lambda s: sorted({a for sub in s for a in sub})),
        'citations':       grouped['citations'].apply(lambda s: sorted({c for sub in s for c in sub})),
    }).reset_index(drop=True)
    if verbose:
        print('Done!')

        # 7) vectorized “unmerge” via prefix‐mask + explode
    #    a) define the prefix test
    def all_prefix(names):
        # 1) normalize: lowercase & strip leading/trailing dots
        cleaned = [n.lower().strip('.') for n in names]
        # 2) dedupe & sort ascending by length
        cleaned = sorted(set(cleaned), key=len)
        # 3) ensure each name is a prefix of the very next one
        return all(
            cleaned[i+1].startswith(cleaned[i])
            for i in range(len(cleaned)-1)
        )

    #    b) compute mask of clusters to keep
    merged['keep'] = merged['all_first_names'].progress_apply(all_prefix)
    
    #    c) keep the good clusters
    good = merged[merged['keep']].drop(columns=['keep'])

    #    d) break the bad clusters into singletons
    bad = merged[~merged['keep']]
    bad_ids = bad['members'].explode().astype(int)
    
    #    e) pull originals for those singletons
    orig = df2.set_index('unique_author_id')
    singles = orig.loc[bad_ids].reset_index()

    #    f) rename & wrap into final schema
    singles = singles.rename(columns={
        'first_name': 'all_first_names',
        'surname':    'all_surnames',
        'id':         'paper_ids'
    })
    for col in ('all_first_names','all_surnames','paper_ids'):
        singles[col] = singles[col].apply(lambda x: [x] if not isinstance(x, list) else x)
    singles['members']          = singles['unique_author_id'].apply(lambda x: [x])
    singles['merged_author_id'] = singles['unique_author_id']
    singles = singles[good.columns]  # align column order

    #    g) concat and return
    return pd.concat([good, singles], ignore_index=True)


def make_hyperedges(df, ):
    """
    Make a hypergraph from the dataframe.
    """
    # explode the papers column
    exploded_papers = df.explode('paper_ids')
    # Group by paper_id to create hyperedges from author_ids 
    hyperedges = exploded_papers.groupby('paper_ids')['merged_author_id'].agg(lambda x: ','.join(map(str, x)))
        
    return hyperedges


def merge_gender_data(df, gender_file:str = 'data/real_world/aps/gendered_names.csv'):
    """
    Merge gender data into the dataframe.
    """
    gendered_names = pd.read_csv(gender_file)
    # Explode the first names column
    exploded_df = df.explode('all_first_names')
    # Merge with gendered names
    merged_df = exploded_df.merge(gendered_names, 
                                 left_on='all_first_names',
                                 right_on='first_name',
                                 how='left')
    
    # Group back by author_id and aggregate the gender data into lists
    gender_cols = ['genderapi_gender', 'genderapi_p', 'genderapi_n', 
              'genderizeio_gender', 'genderizeio_p', 'genderizeio_n',
              'genderapi_p_female', 'genderizeio_p_female']

    agg_dict = {col: list for col in gender_cols}
    agg_dict.update({
        'all_surnames': 'first',
        'all_first_names': tuple,
        'paper_ids': 'first'
    })

    recombined_df = merged_df.groupby('merged_author_id').agg(agg_dict).reset_index()

    recombined_df['genderapi_p_female'] = recombined_df['genderapi_p_female'].progress_apply(lambda vals: vals[np.argmax(abs(np.array(vals) - 0.5))])
    recombined_df['genderizeio_p_female'] = recombined_df['genderizeio_p_female'].progress_apply(lambda vals: vals[np.argmax(abs(np.array(vals) - 0.5))])

    return recombined_df
    

def main():
    # format the aps data
    delimiter= ';\t'
    data_path = 'data/real_world/aps/aps-dataset-metadata-2022/'
    meta = format_aps_data(data_path, delimiter=delimiter)
    meta = pd.read_csv('data/real_world/aps/metadata_raw.csv', sep=delimiter, engine='python')
    meta = meta.rename(columns={'author': 'author_name', 'firstname': 'first_name', 'institutions': 'affiliations'})

    df = format_dataframe(meta)
    df.to_pickle('data/real_world/aps/aps_checkpoint.pkl')

    # Read checkpoint
    df = pd.read_pickle('data/real_world/aps/aps_checkpoint.pkl')

    merged_df = merge_authors(df, verbose=True)
    merged_df.to_pickle('data/real_world/aps/aps_checkpoint_merged.pkl')

    # Read merged dataframe
    print('reading merged dataframe')
    merged_df = pd.read_pickle('data/real_world/aps/aps_checkpoint_merged.pkl')

    print(merged_df.shape)
    print("Max number of first names in a list:", merged_df['all_first_names'].apply(len).max())
    print("Row with most first names:", merged_df.iloc[df['all_first_names'].apply(len).idxmax()]['all_first_names'])
    print(merged_df.columns)

    # Make hypergraph
    hyperedges = make_hyperedges(merged_df)
    hyperedges.to_csv('data/real_world/aps/aps.csv', header=['hyperedges'], index=False)

    # Merge gender data
    merged_df = merge_gender_data(merged_df)

    #output the merged dataframe
    merged_df.to_csv('data/real_world/aps/metadata.csv', index=False)
   
if __name__ == "__main__":
    main()
