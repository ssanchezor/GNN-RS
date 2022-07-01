import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pareto


def Popularity_Graphic(total_recommendations_dict, Title):
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(8, 4))

    plt.title(r'Item Recommendation ' + Title)
    plt.xlabel('Items ranked by popularity')
    plt.ylabel('Number of recommendations')
    dict_order = dict(sorted(total_recommendations_dict.items(), key=lambda item: item[1], reverse=True))
    list_dict = list(dict_order.values())
    print('max num recommendations: ', list_dict[0])
    list_index = list(dict_order.keys())
    list_accumsum = np.cumsum(list_dict)

    total_recommendations = list_accumsum[len(list_accumsum) - 1]
    print('Total Recommendations: ', total_recommendations)
    z = 0
    for i in list_accumsum:
        if i >= 0.8 * total_recommendations:
            break
        else:
            z = z + 1

    print(Title)
    print('Number Products at what reach the 80% of the recommendations', z, '(Accumulate recommendations',
          list_accumsum[z], ')')
    print('Products:{0:2.2f}% '.format((z / len(list_dict)) * 100))
    print(list_index[:10])
    head_tail_split = z
    plt.plot(range(head_tail_split), list_dict[:head_tail_split], alpha=0.8, label=r'Head tail')
    plt.plot(range(head_tail_split, len(list_index)), list_dict[head_tail_split:], label=r'Long tail')
    plt.plot(head_tail_split, list_dict[head_tail_split], '-gD', label='80% of recommendations done')
    plt.axhline(y=list_dict[head_tail_split], linestyle='--', lw=1, c='grey')
    plt.axvline(x=head_tail_split, linestyle='--', lw=1, c='grey')
    plt.xlim([-5, len(list_index)])
    plt.ylim([-5, list_dict[0]])
    plt.legend()
    plt.tight_layout()
    fname = "../data/" + Title + ".pdf"
    plt.savefig(fname)
