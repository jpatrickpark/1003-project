import json
import pickle

import numpy as np
import pandas as pd
from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
import random

from collections import defaultdict
from functools import partial
from itertools import combinations

from utils import invert_dict

def create_co_occurrence_matrix(paper_paper_dict):
    result = defaultdict(partial(defaultdict, int))
    for key, value in paper_paper_dict.items():
        for first, second in combinations(value, 2):
            result[first][second] += 1
            result[second][first] += 1
    result.default_factory = None
    for key in result:
        result[key].default_factory = None
    return result

def delete_those_citing_none(data):
    freesouls = set()
    for user in data:
        if len(data[user])==0:
            freesouls.add(user)
    for user in freesouls:
        del data[user]


def create_paper_paper_dict(debug=False, datadir='../dblp-ref'):
    # It takes about 6 minutes 20 seconds on crunchy5
    if debug:
        DBLP_LIST = [ datadir+'/dblp-ref-3.json' ]
    else:
        DBLP_LIST = [ datadir+'/dblp-ref-0.json',
        datadir+'/dblp-ref-1.json',
        datadir+'/dblp-ref-2.json',
        datadir+'/dblp-ref-3.json' ]

    result = dict()

    for data in DBLP_LIST:
        with open(data) as f:
            line = f.readline()
            while line:
                data = json.loads(line)
                try:
                    result[data["id"]] = data["references"]
                except KeyError:
                    result[data["id"]] = []
                line = f.readline()

    delete_those_citing_none(result)
    return result


def create_user_paper_dict(debug=False, datadir='../dblp-ref'):
    if debug:
        DBLP_LIST = [ datadir+'/dblp-ref-3.json' ]
    else:
        DBLP_LIST = [ datadir+'/dblp-ref-0.json',
        datadir+'/dblp-ref-1.json',
        datadir+'/dblp-ref-2.json',
        datadir+'/dblp-ref-3.json' ]

    result = defaultdict(partial(defaultdict, int))

    for data in DBLP_LIST:
        with open(data) as f:
            line = f.readline()
            while line:
                data = json.loads(line)
                try:  # it happens that some paper does not have "authors" entry.
                    for author in data["authors"]: # assuming this won't error
                        try:
                            for paper in data["references"]:
                                result[author][paper] += 1
                        except KeyError:
                            result[author] # this line creates an entry in result
                except KeyError:
                    # If no authors entry, skip this line.
                    pass
                line = f.readline()
    result.default_factory = None
    for key in result:
        result[key].default_factory = None

    delete_those_citing_none(result)
    return result


def save_pickle(save_filename, obj):
    with open(save_filename, 'wb') as handle:
        return pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(save_filename):
    with open(save_filename, 'rb') as handle:
        return pickle.load(handle)

def save_paper_paper_dict(datadir='../dblp-ref'):
    # resulting file size about 1G
    # Loading this pickle file actually takes much longer than creating the dictionary!
    # It took me 20 minutes and it still did not finish.
    # I recommend just creating the dict without saving it as pickle.
    save_pickle(datadir+'/paper_paper_dict.pickle', create_paper_paper_dict())

def save_numbering_and_reverse(datadir='../dblp-ref'):
    # TODO: add logic so that only creates these files when they don't exist
    numbering, reverse = assign_number_to_paper_id()
    save_pickle(datadir+'/numbering.pickle', numbering)
    save_pickle(datadir+'/reverse.pickle', reverse)

def assign_number_to_paper_id():
    # When using surprise, it seems like we don't need this method.
    # This method assigns nonnegative integer numbers to each paper id,
    # and record it in a dictionary for efficient lookup.
    numbering, reverse = dict(), dict()
    current_id = 0

    for data in DBLP_LIST:
        with open(data) as f:
            line = f.readline()
            while line:
                data = json.loads(line)
                numbering[data["id"]] = current_id
                reverse[current_id]   = data["id"]
                current_id += 1
                line = f.readline()

    return numbering, reverse

def create_surprise_paper_paper_data(paper_paper_dict, add_random_0_entries=False):
    itemList, userList, ratingList = [], [], []

    all_keys_set = set(paper_paper_dict.keys())
    for key, value in paper_paper_dict.items():
        for paper in value:
            itemList.append(paper)
            userList.append(key)
            ratingList.append(1) # "rating" is always 1 for each citation

        # JP 03/28/18 First attempt on trying to add some (not all) entries with 0 ratings
        if add_random_0_entries and len(value)!=0:
            # create candidate set which does not include references
            zero_rating_set_for_key = all_keys_set - set(value)
            # add randomly selected 0 entry, the same number as 1 entries
            for paper in random.sample(zero_rating_set_for_key, len(value)):
                itemList.append(paper)
                userList.append(key)
                ratingList.append(0) # we add 0 entries (no citation) with certain probability

    ratings_dict = {'itemID': itemList, 'userID': userList, 'rating': ratingList}
    df = pd.DataFrame(ratings_dict)

    reader = Reader(rating_scale=(0,1)) # JP: rating scale is 0 (not cited) and 1 (cited)

    data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

    return data

def normalize_user_paper_data(user_paper_dict, rating_scale):
    min_rating = rating_scale[0]
    max_rating = rating_scale[1]

    # initialize max_rating for each paper
    papers_max_ratings = {}
    for papers in user_paper_dict.values():
        for paper in papers.keys():
            papers_max_ratings[paper] = 0

    # find max_rating for each paper
    all_authors_set = set(user_paper_dict.keys())
    for author in all_authors_set:
        all_papers_set = set(user_paper_dict[author].keys())
        for paper in all_papers_set:
            if user_paper_dict[author][paper] > papers_max_ratings[paper]:
                papers_max_ratings[paper] = user_paper_dict[author][paper]

    for author in all_authors_set:
        all_papers_set = set(user_paper_dict[author].keys())
        for paper in all_papers_set:
            y = papers_max_ratings[paper]
            if y == 1:
                continue
            else:
                z = user_paper_dict[author][paper]
                value = (y + max_rating*z - (max_rating+z))/(y-1)
                user_paper_dict[author][paper] = value

    return user_paper_dict


def create_surprise_user_paper_data(user_paper_dict, rating_scale):

    itemList, userList, ratingList = [], [], []

    all_authors_set = set(user_paper_dict.keys())
    for author in all_authors_set:
        all_papers_set = set(user_paper_dict[author].keys())
        for paper in all_papers_set:
            itemList.append(paper)
            userList.append(author)
            ratingList.append(user_paper_dict[author][paper])

    ratings_dict = {'itemID': itemList, 'userID': userList, 'rating': ratingList}
    df = pd.DataFrame(ratings_dict)
    reader = Reader(rating_scale=rating_scale)
    data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

    return data


def create_random_subset_paper_paper_data(size=100000, seed=1003, debug=False, datadir='../dblp-ref'):
    #Build a random subset of dictionary, where we only retain references to themselves
    mydict = create_paper_paper_dict(debug=debug,datadir=datadir)
    if size > len(mydict):
        size = len(mydict)
    random.seed(seed)
    random_dict = {k: list(mydict[k]) for k in random.sample(mydict.keys(),size)}
    for each in random_dict:
            for ref in mydict[each]:
                if ref not in random_dict:
                    random_dict[each].remove(ref)
    delete_those_citing_none(random_dict)
    return random_dict

def create_random_subset_user_paper_data(size=5000, seed=1003, debug=False, datadir='../dblp-ref'):
    #Build a random subset of dictionary, where we only retain references to themselves
    mydict = create_user_paper_dict(debug=debug,datadir=datadir)
    if size > len(mydict):
        size = len(mydict)
    random.seed(seed)
    random_dict = {k: mydict[k] for k in random.sample(mydict.keys(),size)}

    # DYK: I believe for user_paper_dict, we don't need to delete these.
    # for user in random_dict:
    #     for paper in mydict[user]:
    #         if ref not in random_dict:
    #             random_dict[each].remove(ref)
    return random_dict

def paper_paper_train_test_split(user_paper_dict, test_size = .11):
    if test_size > .1:
        test_size = .1
    all_nonzero_entries = set()
    users_to_be_prevented = set()
    papers_to_be_prevented = set()

    for user in user_paper_dict:
        if len(user_paper_dict[user]) < 2:
            users_to_be_prevented.add(user)
        for paper in user_paper_dict[user]:
            all_nonzero_entries.add((user,paper))
            cited_by_how_many = 0
            for userrr in user_paper_dict:
                if paper in user_paper_dict[userrr]:
                    cited_by_how_many += 1
            if cited_by_how_many < 2:
                papers_to_be_prevented.add(paper)

    entries_to_be_prevented = set()
    for entry in all_nonzero_entries:
        if entry[0] in users_to_be_prevented or entry[1] in papers_to_be_prevented:
            entries_to_be_prevented.add(entry)

    splittables = all_nonzero_entries - entries_to_be_prevented
    splittable_dict = defaultdict(list)
    for entry in splittables:
        splittable_dict[entry[0]].append(entry[1])

    L = len(all_nonzero_entries)
    M = len(entries_to_be_prevented)
    x = test_size/(1 - M/L)

    testset = defaultdict(list)
    for user in splittable_dict:        ######## Find this value!!!!
        if len(splittable_dict[user]) > 1:  # data <= 2 are approx. 4%
            rand_sample = random.sample(splittable_dict[user], int(np.ceil(x*len(splittable_dict[user]))))
            for paper in rand_sample:
                testset[user].append(paper)
    testset.default_factory = None

    trainset = defaultdict(list)
    for entry in splittables:
        try:
            papers = testset[entry[0]]
            if entry[1] not in papers:
                trainset[entry[0]].append(entry[1])
        except KeyError:
            trainset[entry[0]].append(entry[1])
    for entry in entries_to_be_prevented:
        trainset[entry[0]].append(entry[1])
    trainset.default_factory = None

    return trainset, testset


def user_paper_train_test_split(user_paper_dict, test_size = .25):
    if test_size > .25:
        test_size = .25
    all_nonzero_entries = set()
    users_to_be_prevented = set()
    papers_to_be_prevented = set()
    entries_to_be_prevented = set()

    for user in user_paper_dict:
        if len(user_paper_dict[user]) < 2:
            users_to_be_prevented.add(user)
        for paper in user_paper_dict[user]:
            all_nonzero_entries.add((user,paper,user_paper_dict[user][paper]))
            cited_by_how_many = 0
            for userrr in user_paper_dict:
                if paper in user_paper_dict[userrr]:
                    cited_by_how_many += 1
            if cited_by_how_many < 2:
                papers_to_be_prevented.add(paper)

    for entry in all_nonzero_entries:
        if entry[0] in users_to_be_prevented or entry[1] in papers_to_be_prevented:
            entries_to_be_prevented.add(entry)

    splittables = all_nonzero_entries - entries_to_be_prevented
    splittable_dict = defaultdict(partial(defaultdict, int))
    for entry in splittables:
        splittable_dict[entry[0]][entry[1]] = entry[2]

    L = len(all_nonzero_entries)
    M = len(entries_to_be_prevented)
    x = test_size/(1 - M/L)

    testset = defaultdict(partial(defaultdict, int))
    for user in splittable_dict:
        if len(splittable_dict[user]) > 4:  # data <= 5 are approx. 2%

            rand_sample = random.sample(splittable_dict[user].keys(),int(np.ceil(x*len(splittable_dict[user]))))
            for paper in rand_sample:
                testset[user][paper] = splittable_dict[user][paper]
    testset.default_factory = None
    for key in testset:
        testset[key].default_factory = None

    trainset = defaultdict(partial(defaultdict, int))
    for entry in splittables:
        try:
            testset[entry[0]]
            try:
                testset[entry[0]][entry[1]]
            except KeyError:
                trainset[entry[0]][entry[1]] = entry[2]
        except KeyError:
            trainset[entry[0]][entry[1]] = entry[2]
    for entry in entries_to_be_prevented:
        trainset[entry[0]][entry[1]] = entry[2]
    trainset.default_factory = None
    for key in trainset:
        trainset[key].default_factory = None

    return trainset, testset

def create_train_test_dic(total_dic):
    testdic  = defaultdict(list)
    traindic = defaultdict(list)

    invert_total_dic = invert_dict(total_dic)
    for user in total_dic:
        if len(total_dic[user]) < 2:
            traindic[user] = total_dic[user]
        else:
            i = 0
            for ref in total_dic[user]:
                i = i+1
                if i < 2:
                    traindic[user].append(ref)
                else:
                    if len(invert_total_dic[ref]) < 2:
                        traindic[user].append(ref)
                    else:
                        invert_total_dic[ref].remove(user)
                        testdic[user].append(ref)

    return traindic, testdic

if __name__ == "__main__":
    create_surprise_paper_paper_data(create_random_subset_paper_paper_data(100000,debug=True),True)
