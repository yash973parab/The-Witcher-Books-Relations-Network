import pandas as pd
import numpy as np
import spacy
from spacy import displacy
import networkx as nx
import matplotlib.pyplot as plt
#==============================================================================================================#
# IMPORT ENGLISH LANGUAGE MODEL FROM SPACY
#!python3 -m spacy download en_core_web_sm
# the above line is jupyter or google collab specific file, so it wont work in python script window, use terminal

# LOAD SPACY ENGLISH LANGUAGE MODEL
NER = spacy.load("en_core_web_sm")

#==============================================================================================================#
# LOAD BOOKS TEXT
import os
# get all book files in the data directory
path_to_data = r"D:\Self Upskilling\PythonProjects\Witcher Project\the_witcher_network-main\data"
all_books = [b for b in os.scandir(path_to_data) if '.txt' in b.name]
print(all_books)

book = all_books[1]
book_text = open(book).read()
book_doc = NER(book_text)

# visualise identified entities
#displacy.serve(book_doc[0:2000], style="ent")
# this will show in Run - Serving on http://0.0.0.0:5000
# go to your web browser - chrome or firefox - http://127.0.0.1:5000/ ...here you will find the color coded nlp of spacy

#==============================================================================================================#
# LOAD CHARACTERS
# read characters
character_df = pd.read_csv(r"D:\Self Upskilling\PythonProjects\Witcher Project\the_witcher_network-main\characters.csv")
# remove brackets and text within brackets
import re
character_df['character'] = character_df['character'].apply(lambda x: re.sub(r"[\(].*?[\)]", "", x))
character_df['character_firstname'] = character_df['character'].apply(lambda x: x.split(' ', 1)[0])
#print(character_df)

#==============================================================================================================#
# GET NAMED ENTITY LIST PER SENTENCE
sent_entity_df = []
# loop through sentences, store named entity list for each sentence
for sent in book_doc.sents:
    entity_list = [ent.text for ent in sent.ents]
    sent_entity_df.append({"sentence": sent, "entities": entity_list})

sent_entity_df = pd.DataFrame(sent_entity_df)
#print(sent_entity_df)

# function to filter out non-character entities
def filter_entity(ent_list, character_df):
    return [ent for ent in ent_list
            if ent in list(character_df.character)
            or ent in list(character_df.character_firstname)]
filter_entity(["Geralt", "Yash", "2"], character_df)

sent_entity_df['character_entities'] = sent_entity_df['entities'].apply(lambda x: filter_entity(x, character_df))
# filter or remove sentences which don't have any character entities
sent_entity_df_filtered = sent_entity_df[sent_entity_df['character_entities'].map(len) > 0].copy()
sent_entity_df_filtered.head(10)

# print(sent_entity_df_filtered)
# take only first name of characters
sent_entity_df_filtered['character_entities'] = sent_entity_df_filtered['character_entities'].apply(lambda x:
                                                                                                    [item.split()[0]
                                                                                                     for item in x])
pd.reset_option('^display.', silent=True)
#print(sent_entity_df_filtered)

#==============================================================================================================#
# CREATE RELATIONSHIP
window_size = 5
relationships = []

for i in range(sent_entity_df_filtered.index[-1]):
    end_i = min(i+5, sent_entity_df_filtered.index[-1])
    char_list = sum((sent_entity_df_filtered.loc[i:end_i].character_entities), [])

    # Remove duplicated characters that are next to each other
    char_unique = [char_list[i] for i in range(len(char_list))
                   if (i==0) or char_list[i] != char_list[i-1]]

    if len(char_unique) > 1:
        for idx, a in enumerate(char_unique[:-1]):
            b = char_unique[idx + 1]
            relationships.append({"source":a, "target":b})

relationships_df = pd.DataFrame(relationships)
pd.set_option('display.max_rows', None)
#print(relationships_df)

# sort the cases with a->b and b->a
relationships_df = pd.DataFrame(np.sort(relationships_df.values, axis = 1), columns = relationships_df.columns)
relationships_df['value'] = 1
relationships_df = relationships_df.groupby(['source','target'], sort=False).sum().reset_index()
#print(relationships_df.head(10))

#==============================================================================================================#
# GRAPH ANALYSIS and VISUALIZATION
# create a graph from a pandas dataframe
G = nx.from_pandas_edgelist(relationships_df,
                            source = 'source',
                            target = 'target',
                            edge_attr = 'value',
                            create_using=nx.Graph())
# filter and create new graph where source has met more than 5 times to the target
G_filtered = nx.Graph(((u, v, e) for u, v, e in G.edges(data=True) if e['value'] > 5))

# Graph visualization - Networkx
#pos = nx.kamada_kawai_layout(G)
#nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)
#plt.show()

# Graph visualizatio - Pyvis
from pyvis.network import Network
net = Network(notebook = True, width = "1000px", height = "800px", bgcolor = "#222222", font_color = "white")
node_degree = dict(G.degree)

# setting up node size attribute
nx.set_node_attributes(G, node_degree, 'size')
net.from_nx(G)
#net.show("Witcher.html")

# degree centrality
degree_dict = nx.degree_centrality(G)
degree_df = pd.DataFrame.from_dict(degree_dict, orient='index', columns=['centrality'])
# plot top 10 nodes
degree_df.sort_values(by=['centrality'], ascending=False)[0:9].plot(kind='bar')

# betweenness centrality
betweenness_dict = nx.betweenness_centrality(G)
betweenness_df = pd.DataFrame.from_dict(betweenness_dict, orient='index', columns=['centrality'])
# plot top 10 nodes
betweenness_df.sort_values(by=['centrality'], ascending=False)[0:9].plot(kind='bar')

# closeness centrality
closeness_dict = nx.betweenness_centrality(G)
closeness_df = pd.DataFrame.from_dict(closeness_dict, orient='index', columns=['centrality'])
# plot top 10 nodes
closeness_df.sort_values(by=['centrality'], ascending=False)[0:9].plot(kind='bar')

# save centrality measures
#nx.set_node_attributes(G, degree_dict, 'degree_centrality')
#nx.set_node_attributes(G, betweenness_dict, 'betweenness_centrality')
#nx.set_node_attributes(G, closeness_dict, 'closeness_centrality')

#==============================================================================================================#
# COMMUNITY DETECTION
import community as community_louvain
communities = community_louvain.best_partition(G)
nx.set_node_attributes(G, communities, 'group')
com_net = Network(notebook = True, width = "100%", height = "800px", bgcolor = "#222222",
                  font_color = "white", select_menu=True)
com_net.from_nx(G)
#com_net.from_nx(G_filtered)
com_net.show_buttons(filter_=['physics'])
com_net.show("Witcher_Communities.html")