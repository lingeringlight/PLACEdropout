import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
import matplotlib.lines as mlines


COLORS = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'white']
marker_types = ['.', 'v', 's', '*', 'p', 'H', 'X', '1', '8', ]

def show_scatter(tsne_X, label, marker_size=2, marker_type='o', imgs=None, ax=None, texts=None):
    # imgs : 3 x H x W
    # label : N
    # tsne_X : N x 2
    if ax is None:
        ax = plt.gca()

    label = np.array(label)

    if imgs is None:
        # colors = np.array(sns.color_palette("husl", label.max()+1))
        # plt.scatter(tsne_X[:, 0], tsne_X[:, 1], color=plt.cm.Set1(label), s=marker_size, marker=marker_type)

        color_num = np.unique(label).max() + 1
        if color_num > 8:
            print('ranbow colors')
            cm = plt.get_cmap('gist_rainbow')
            colors = np.array([cm(1. * i / color_num) for i in range(color_num)])[label]
        else:
            label[label == 5] = 7  # 5太黄了
            colors = plt.cm.Set1(label)
        ret = plt.scatter(tsne_X[:, 0], tsne_X[:, 1], color=colors, s=marker_size,  marker=marker_type)
    else:
        imgs = imgs.swapaxes(1, 2).swapaxes(2, 3)  # to channel last
        for i, (img, (x0, y0)) in enumerate(zip(imgs, tsne_X)):
            img = ((img * np.array([.229, .224, .225]).reshape(1, 1, 3) + np.array([.485, .456, .406]).reshape(1, 1, 3)) * 255).astype(np.int32)
            img = OffsetImage(img, zoom=0.2)  # 224*0.2 =
            ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
            ax.add_artist(ab)
            if texts is not None:
                offsetbox = TextArea(str(texts[i]), textprops=dict(alpha=1.0, size="smaller"))
                ab = AnnotationBbox(offsetbox, (x0, y0),
                                    xybox=(0, -27),
                                    xycoords='data',
                                    boxcoords="offset points",
                                    # arrowprops=dict(arrowstyle="->")
                                    )
                ax.add_artist(ab)
    return ret


class tSNE():
    @staticmethod
    def get_tsne_result(X, metric='euclidean', perplexity=30):
        """  Get 2D t-SNE result with sklearn

        :param X: feature with size of N x C
        :param metric: 'cosine', 'euclidean', and so on.
        :param perplexity:  the preserved local structure size
        """
        try:
            from sklearn.manifold.t_sne import TSNE
        except Exception as e:
            from sklearn.manifold._t_sne import TSNE
        tsne = TSNE(n_components=2, metric=metric, perplexity=perplexity)
        tsne_X = tsne.fit_transform(X)
        tsne_X = (tsne_X - tsne_X.min()) / (tsne_X.max() - tsne_X.min())
        return tsne_X

    @staticmethod
    def plot_tsne(tsne_X, labels, domain_labels=None, imgs=None, texts=None, save_name=None, figsize=(10, 10), marker_size=20, label_name=None):
        """ plot t-SNE results. All parameters are numpy format.

        Args:
            tsne_X: N x 2
            labels: N
            domain_labels: N
            imgs: N x 3 x H x W
            save_name: str
            figsize: tuple of figure size
            marker_size: size of markers
        """
        plt.figure(figsize=figsize)
        scatters = []
        if domain_labels is not None:
            # plot each domain with different shape of markers
            domains = np.unique(domain_labels)
            for d in domains:
                idx = domain_labels == d
                x_tmp = imgs[idx] if imgs is not None else None
                text_tmp = texts[idx] if texts is not None else None
                scatter = show_scatter(tsne_X[idx], labels[idx], marker_size=marker_size, marker_type=marker_types[d], imgs=x_tmp, texts=text_tmp)
                scatters.append(scatter)
        else:
            # plot simple clusters of classes with different colors
            show_scatter(tsne_X, labels, marker_size=marker_size, marker_type=marker_types[0], imgs=imgs, texts=texts)

        # plot legend
        each_labels = np.unique(labels)
        legend_elements = []
        for l in each_labels:
            if label_name is not None:
                L = label_name[l]
            else:
                L = str(l)
            legend_elements.append(mlines.Line2D([0], [0], marker='o', color='w', label=L, markerfacecolor=plt.cm.Set1(l), markersize=5))
        legend2 = plt.legend(handles=legend_elements, loc='upper right')
        plt.gca().add_artist(legend2)

        domain_names = ['Photo', 'Art', 'Sketch', 'Cartoon']
        legend1 = plt.legend(scatters, domain_names, loc='upper left')
        plt.gca().add_artist(legend1)

        if save_name is not None:
            plt.savefig(save_name, bbox_inches='tight', dpi=500)
        # plt.show()