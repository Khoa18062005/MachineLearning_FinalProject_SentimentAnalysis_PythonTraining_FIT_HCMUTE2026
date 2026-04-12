import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import accuracy_score
from app.core.config import settings
matplotlib.use('Agg')
class AblationService:
    @staticmethod
    def calculate_reduced_vote(preds_list):
        """Logic bỏ phiếu đa số cho nhóm 2 hoặc 3 model"""
        stacked_preds = np.stack(preds_list, axis=1)
        n_models = len(preds_list)
        # Nếu 3 model: cần >= 2 phiếu. Nếu 2 model: cần >= 2 phiếu (đồng thuận).
        threshold = (n_models // 2) + 1
        votes_for_4 = np.sum(stacked_preds == 4, axis=1)
        return np.where(votes_for_4 >= threshold, 4, 0)

    @staticmethod
    def calculate_dual_vote_logic(custom_votes, library_votes):
        """Mô phỏng Track-Dual Validation: Nếu lệch nhau, ưu tiên Library"""
        dual_preds = []
        for c, l in zip(custom_votes, library_votes):
            dual_preds.append(c if c == l else l)
        return np.array(dual_preds)

    @classmethod
    def run_ablation_and_save_chart(cls, y_true, preds_mnb, preds_svm, preds_xgb, group_id="custom"):
        """Xử lý cho nhóm đơn (Custom hoặc Library)"""
        scenarios = {
            "Gốc": [preds_mnb, preds_svm, preds_xgb],
            "Tắt MNB (SVM + XGB)": [preds_svm, preds_xgb],
            "Tắt SVM (MNB + XGB)": [preds_mnb, preds_xgb],
            "Tắt XGBoost (MNB + SVM)": [preds_mnb, preds_svm]
        }
        results = {name: accuracy_score(y_true, cls.calculate_reduced_vote(models)) * 100
                   for name, models in scenarios.items()}
        return cls._draw_chart(results, group_id)

    @classmethod
    def run_dual_ablation_and_save_chart(cls, y_true, custom_dict, library_dict):
        """Xử lý đặc biệt cho nhóm Dual: Tắt đồng thời ở cả 2 phía"""
        scenarios = [
            ("Gốc", ["mnb", "svm", "xgb"]),
            ("Tắt MNB (SVM + XGBoots)", ["svm", "xgb"]),
            ("Tắt SVM (MNB + XGBoots)", ["mnb", "xgb"]),
            ("Tắt XGBoost (MNB + SVM)", ["mnb", "svm"])
        ]
        results = {}
        for title, active_models in scenarios:
            # Tính vote cho từng nhóm
            vote_c = cls.calculate_reduced_vote([custom_dict[m] for m in active_models])
            vote_l = cls.calculate_reduced_vote([library_dict[m] for m in active_models])
            # Chốt hạ bằng logic Dual
            final_dual = cls.calculate_dual_vote_logic(vote_c, vote_l)
            results[title] = accuracy_score(y_true, final_dual) * 100

        return cls._draw_chart(results, "dual")

    @staticmethod
    def _draw_chart(ablation_results, group_id):
        labels = list(ablation_results.keys())
        accuracies = list(ablation_results.values())

        # Màu sắc theo đúng phong cách Dashboard của bạn
        colors = ['#ef4444' if "Gốc" in l else '#2A95BF' for l in labels]

        plt.figure(figsize=(16, 9), facecolor='white')
        fig = plt.gcf()
        fig.patch.set_alpha(0)

        rect = patches.FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
                                      boxstyle="round,pad=0.01,rounding_size=0.04",
                                      facecolor="white", edgecolor="none",
                                      transform=fig.transFigure, zorder=-1)
        fig.patches.append(rect)

        ax = plt.gca()
        ax.set_facecolor('none')
        bars = plt.bar(labels, accuracies, color=colors, alpha=0.85, width=0.6, zorder=3)

        title_map = {"custom": "Nhóm CUSTOM", "library": "Nhóm LIBRARY", "dual": "Nhóm DUAL (Track-Dual)"}
        plt.title(f'Ablation Study: Tầm quan trọng của thành phần trong {title_map.get(group_id)}',
                  fontsize=18, pad=20, fontweight='bold', color='#1e293b')

        plt.ylim(max(0, min(accuracies) - 5), min(100, max(accuracies) + 5))
        plt.ylabel('Accuracy (%)', fontsize=14, color='#475569')
        plt.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
        plt.xticks(fontsize=15, fontweight='bold', color='#1e293b')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.2,
                     f'{height:.2f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout(pad=3.0)

        file_name = f"ablation_study_{group_id}.png"
        plt.savefig(os.path.join(settings.BASE_DIR, file_name), dpi=300, transparent=True)
        plt.close()
        return file_name