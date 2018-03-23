import json
import pickle

data_list = [ 'dblp-ref/dblp-ref-0.json',
'dblp-ref/dblp-ref-1.json',
'dblp-ref/dblp-ref-2.json',
'dblp-ref/dblp-ref-3.json' ]

#data_list = [ 'dblp-ref/dblp-ref-3.json' ]

def create_paper_paper_dict():
    # It takes about 6 minutes 20 seconds on crunchy5
    result = dict()

    for data in data_list:
        with open(data) as f:
            line = f.readline()
            while line:
                data = json.loads(line)
                try:
                    result[data["id"]] = data["references"]
                except KeyError:
                    result[data["id"]] = []
                line = f.readline()

    return result

def save_pickle(save_filename, obj):
    with open(save_filename, 'wb') as handle:
        return pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(save_filename):
    with open(save_filename, 'rb') as handle:
        return pickle.load(handle)

def save_paper_paper_dict():
    # resulting file size about 1G
    # Loading this pickle file actually takes much longer than creating the dictionary!
    # It took me 20 minutes and it still did not finish.
    save_pickle('dblp-ref/paper_paper_dict.pickle', create_paper_paper_dict())

def save_numbering_and_reverse():
    # TODO: add logic so that only creates these files when they don't exist
    numbering, reverse = assign_number_to_paper_id()
    save_pickle('dblp-ref/numbering.pickle', numbering)
    save_pickle('dblp-ref/reverse.pickle', reverse)

def assign_number_to_paper_id():
    # When we create a huge matrix,
    # we want to have a way to figure out matrix index for a given paper id.
    # We assign nonnegative integer numbers to each paper id,
    # and record it in a dictionary for efficient lookup.
    numbering, reverse = dict(), dict()
    current_id = 0

    for data in data_list:
        with open(data) as f:
            line = f.readline()
            while line:
                data = json.loads(line)
                numbering[data["id"]] = current_id
                reverse[current_id]   = data["id"]
                current_id += 1
                line = f.readline()

    return numbering, reverse

if __name__ == '__main__':
    #mydict = create_paper_paper_dict()
    #save_numbering_and_reverse()
    numbering = load_pickle('dblp-ref/numbering.pickle')
    reverse = load_pickle('dblp-ref/reverse.pickle')

    #print(len(numbering))
    #print(reverse[79006])
    #reference
