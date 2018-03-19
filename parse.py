from lxml import html
from pprint import pprint as pp
from bs4 import BeautifulSoup as BS
import numpy as np
import pandas as pd
import requests as rq
import time

rejects = ['shop', 'explore', 'store', 'deal', 'store', 'best', 'seller', 'top', 'offer', 'return', 'featured', 'brand', 'new']

def load_tree(url):
    page = rq.get('https://www.amazon.co.uk'+url)
    while page.status_code != 200:
        time.sleep(0.02)
        page = rq.get('https://www.amazon.co.uk'+url)
    tree = BS(page.content, 'html5lib')
    return tree

def extract_brand_browse_page(tree):
    h2 = tree.find('h2', string='Featured Brands')
    if h2 == None:
        return ''
    ul = h2.find_next_sibling('ul')
    if ul != None:
        node = ul.find_all('a')[-1]
        if 'See more' in node.text:
            return node.get('href')
        else:
            import re
            h = tree.find('a', href=re.compile('pickerToList')).get('href').split('=')[-1]
            return '/gp/search/other?rh='+h+'&pickerToList=lbr_brands_browse-bin'
    else:
        return 'Rejected'

def extract_brand_alphabetical_links(url):
    links = []
    tree = load_tree(url)
    node_id = tree.find('span', {'class': 'pagnLink'}).find('a').get('href').split('=')[-1]
    nodes = tree.find_all('span', {'class': 'pagnLink'})
    for n in nodes:
        letter = n.find('a').text.lower()
        if letter == '#':
            links.append('/gp/search/other/ref=sr_in_1_-2?rh='+node_id+'&pickerToList=lbr_brands_browse-bin&indexField=%23')
        else:
            links.append('/gp/search/other/ref=sr_in_'+letter+'_-2?rh='+node_id+'&pickerToList=lbr_brands_browse-bin&indexField='+letter)
    return list(set(links))

def extract_brand_names(url):
    tree = load_tree(url)
    return [_.text for _ in tree.find_all('span', {'class': 'refinementLink'})]

def extract_sub_category_links(depth, url):
    struct = {}
    tree = load_tree(url)
    bbp = extract_brand_browse_page(tree)
    if bbp == '':
        for h in tree.find_all('h3'):
            if not any(_ in h.text.lower() for _ in rejects):
                struct[h.text] = {}
                ul = h.find_next_sibling('ul')
                if ul != None:
                    for a in h.find_next_sibling('ul').find_all('a'):
                        struct[h.text][a.text] = a.get('href')
    elif bbp == 'Rejected':
        print depth*'\t'+tree.title.text.encode('ascii', 'ignore'), ' : Rejected'
#        return {tree.title.text: 'Rejected'}
    else:
        print(depth*'\t'+tree.title.text.encode('ascii', 'ignore')),
        brands = set()
        for l in extract_brand_alphabetical_links(bbp):
            brands |= set(extract_brand_names(l))
        print ' : ', str(len(brands))
        return sorted(list(brands))
    for k in struct.keys():
        print depth*'\t'+k+'==>'
        for i,j in struct[k].items():
            struct[k][i] = extract_sub_category_links(depth+1, j)
    return struct

def extract_category_links(url):
    tree = load_tree(url)
    struct = {}
    for s in tree.find_all('span', class_='nav-a-content'):
        h = s.find_parent('a').get('href')
        if not any(_ in s.text.lower() for _ in rejects):
            struct[s.text] = h
    return struct

def extract_brands(url):
    cats = extract_category_links(url)
    pp(cats)
    print
    for c,u in cats.items():
        print c
        cats[c] = extract_sub_category_links(1, u)
    return cats

def load_dict_from_file(filename):
    import ast
    with open(filename, 'r') as f:
        d = ast.literal_eval(f.read())
    return d

def create_brand_set(sub_dict):
    if isinstance(sub_dict, list):
        return set(sub_dict)
    out = set()
    for i in sub_dict.keys():
        out |= create_brand_set(sub_dict[i])
    return out

def create_tag_set(sub_dict, start_tag):
    if isinstance(sub_dict, list):
        return [start_tag]
    out = list()
    for i in sub_dict.keys():
        out.extend([start_tag+' -> '+_ for _ in create_tag_set(sub_dict[i], i)])
    return out

def create_tag_map(sub_dict, tag, global_dict):
    if isinstance(sub_dict, list):
        global_dict[tag] = sub_dict
        print tag
        return
    for i in sub_dict.keys():
        create_tag_map(sub_dict[i], tag+'/'+i, global_dict)
    return

def load_all_data(path, filenames):
    all_dicts = {}
    tags = []
    brands = set()
    for g in filenames:
        all_dicts[g] = load_dict_from_file(path + g + '.txt')
        tags.extend(create_tag_set(all_dicts[g], g))
        brands |= create_brand_set(all_dicts[g])
    return all_dicts, tags, brands

def extract_all(scraping_links):
    for g,l in scraping_links.items():
        print g.upper()
        t = extract_brands(l)
        with open(g+'.txt', 'wt') as f:
            pp(t, stream=f)
        print 20*'='

def extend_data(df):
    df1 = df.copy()
    df1['text'] = df1['text'].apply(lambda _: _.upper())
    df = pd.concat([df, df1], axis=0).reset_index()
    return df

def convert_case_and_add(text):
    t = text + ' ' + text.lower() + ' ' + text.upper()
    t += ' ' + ' '.join([x[0].upper()+x[1:] for x in text.split()])
    return t

def sieve_tokens(tokens):
    tokens = sorted(tokens, key=lambda _: len(_))
    sieved = list()
    for i in range(len(tokens)):
        if not any((tokens[i] in _) or (tokens[i].lower() in _) or (tokens[i].lower() in _.lower()) for _ in tokens[i+1:]):
            sieved.append(tokens[i])
        if (len(sieved)+1)%1000 == 0:
            print len(sieved)
    return sieved

scraping_links = {
        'fashion': '/b/ref=topnav_storetab_top_ap_arrow?ie=UTF8&node=11961407031',
        'beauty': '/beauty-cosmetics/b/ref=nav_shopall_bty?ie=UTF8&node=117332031',
        'toys': '/toys/b/ref=topnav_storetab_toys?ie=UTF8&node=468292',
        'sports': '/Sports-Exercise-Fitness-Bikes-Camping/b/ref=topnav_storetab_sg?ie=UTF8&node=318949011',
#        'videogames': '/PC-Video-Games-Consoles-Accessories/b/ref=topnav_storetab_vg_h_?ie=UTF8&node=300703',
        }

#fashion = load_dict_from_file('fashion.txt')
#del fashion['']
#del fashion['KIDS & BABY']["Women's Fashion"]
#del fashion['KIDS & BABY']["Men's Fashion"]
#del fashion['KIDS & BABY']["Luggage & Travel Gear"]
#with open('fashion.txt', 'wt') as f:
#    pp(fashion, stream=f)
#
#sports = load_dict_from_file('sports.txt')
#del sports['Sports & Outdoors']
#with open('sports.txt', 'wt') as f:
#    pp(sports, stream=f)
#
