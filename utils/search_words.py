import tagging_utils as tu

import nltk
nltk.download('all')

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF

# COMMAND ----------

from nltk.corpus import wordnet

# COMMAND ----------

def find_noun_in_phrase(search_item):
    """ Find the noun in a search string using NLTK.
    :param search_item: search string
    :return: str noun
    """

    search_list = search_item.split()
    tagged_sentence = nltk.corpus.treebank.tagged_sents(tagset='universal')
    train_set, test_set = train_test_split(tagged_sentence, test_size=0.2, random_state=1234)

    x, y = tu.prepareData(tagged_sentence)

    crf = CRF(
        algorithm='lbfgs',
        c1=0.01,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )

    crf.fit(x, y)

    sentence_features = [tu.features(search_list, index) for index in range(len(search_list))]
    tagged_words = list(zip(search_list, crf.predict([sentence_features])[0]))

    noun = [word[0] for word in tagged_words if word[1] == 'NOUN']

    noun_str = noun[0]

    return noun_str


def get_data(noun_str, spark):
    """
    Get all product names containing the noun identified from the search string.
    :param sql_context: Spark sqlContext
    :param noun_str: noun string
    :return: Spark data frame
    """

    grocery_items = (
    spark
    .read
    .format("csv")
    .option("header", "true")
    .option("delimiter", ";")
    .load("/content/gdrive/MyDrive/colab_demo/Grocery_UPC_Database.csv")
    )

    grocery_items.registerTempTable("grocery_upc_database")

    result = spark.sql(f"select name "
                             f"from grocery_upc_database "
                             f"WHERE lower(name) like '%{noun_str}%'")
    print("Fetching data.")
    return result


def calculate_similarity_coeff(data, search_list):
    """ Calculate the similarity coefficient between the full product name and the search string.
    :param data: Spark data frame containing the noun that was extracted from the search string
    :param search_list: all items in the search string in the form of a list
    :return: dict with similarity coefficients for every product in the database containing the noun
    """

    data_list = data.select("name").rdd.flatMap(lambda x: x).collect()
    data_list_new = [x.strip() for x in data_list]
    data_list_new = set(data_list)

    coeff_dict = {}

    for i in data_list_new:
        i_list = i.split(" ")
        i_set = set(i_list)

        coeff_dict[i] = 0

        for w1 in search_list:
            w1 = w1.lower()
            w1_coeffs = []
            for w2 in i_set:
                w2 = w2.lower()
                try:
                    d = wordnet.synset(w1 + '.n.01')
                    g = wordnet.synset(w2 + '.n.01')
                    w1_coeffs.append(d.wup_similarity(g))

                except:
                    w1_coeffs.append(0)
                    continue

            coeff_dict[i] += max(w1_coeffs)

    return coeff_dict


def run_calculations(spark, search_item):
    """
    Executes the Spark application demo.
    The application extracts the noun from a search phrase, searches the product database for products with this noun
    and then calculates similarity coefficients to see which is most likely the one meant by the customer.
    This is only a demo and the similarity coefficients do not reflect in-depth analysis and research!
    :param sqlContext:
    :return:
    """
    search_list = search_item.split(" ")
    
    noun_str = find_noun_in_phrase(search_item)

    data = get_data(noun_str.lower(), spark)
    data.cache().count()
#        data.show()

    coeff_dict = calculate_similarity_coeff(data, search_list)

    for w in sorted(coeff_dict, key=coeff_dict.get, reverse=True):
        print(w, coeff_dict[w])
