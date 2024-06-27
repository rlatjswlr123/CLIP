import logging
import torch
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt


# TODO: depth가 없는 class 분류용 tsne 코드 필요
def feature2tsne(features, gt_category, gt_status, epoch, tsne_save_dir):
    category_indices = {}
    for i, category in enumerate(gt_category):
        if category not in category_indices:
            category_indices[category] = [i]
        else:
            category_indices[category].append(i)

    category_gt_status = {}
    category_features = {}
    for category, indices in category_indices.items():
        category_gt_status[category] = [gt_status[idx] for idx in indices]
        category_features[category] = features[indices]

    for category_name, f_feature in category_features.items():
        print(category_name)
        n_samples = f_feature.shape[0]  # 데이터 샘플 수 확인
        perplexity = min(30, n_samples - 1)  # perplexity 값 조정
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        feature_tsne = tsne.fit_transform(f_feature)

        unique_numbers = {status: i for i, status in enumerate(set(category_gt_status[category_name]))}
        status_idx = [unique_numbers[status] for status in category_gt_status[category_name]]
        name_list = unique_numbers.keys()
        fig, ax = plt.subplots()  # 새로운 figure와 axis 생성
        
        # 산점도 그리기
        scatter = ax.scatter(feature_tsne[:, 0], feature_tsne[:, 1], c=status_idx, cmap='tab10')
        
        # legend 만들기
        if name_list != None:    
            handles = [plt.Line2D([0], [0], marker='o', color='w', label=name, markersize=12, markerfacecolor=color) 
                    for name, color in zip(name_list, scatter.to_rgba(np.unique(status_idx)))]
            ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
            
        # 축 범위 조정
        ax.set_xlim(feature_tsne[:, 0].min() - 5, feature_tsne[:, 0].max() + 5)
        ax.set_ylim(feature_tsne[:, 1].min() - 5, feature_tsne[:, 1].max() + 5)
        
        # 제목, 축 레이블 설정
        ax.set_title(f'T-SNE Visualization of {category_name}')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        
        plt.tight_layout()  # 레이아웃 조정
        plt.savefig(f"{tsne_save_dir}/epoch_{epoch}_{category_name}_tsne.jpg")
        plt.close()


# confusion matrix를 예쁘게 출력하는 함수
def print_confusion_matrix(cf_matrix, menu_name):
    print("Confusion Matrix:")
    print("{:<12}".format("Predicted:"), end="\t")
    for name in menu_name:
        print("{:<12}".format(name), end="\t")
    print("\nActual:")
    for i in range(len(menu_name)):
        print("{:<12}".format(menu_name[i]), end="\t")
        for j in range(len(menu_name)):
            print("{:<12}".format(cf_matrix[i][j]), end="\t\t")
        print()