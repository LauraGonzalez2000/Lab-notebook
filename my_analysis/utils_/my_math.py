import pandas as pd
import numpy as np
from scipy.stats import normaltest, levene, f_oneway, tukey_hsd, kruskal, ttest_ind, mannwhitneyu
import scikit_posthocs as sp
import numpy as np
from scipy.stats import normaltest, levene

def test_parametric_conditions(*groups):
    """
    Tests if multiple groups meet parametric assumptions: normality and homoscedasticity.
    
    Parameters:
        *groups (array-like): Two or more data samples.
    
    Returns:
        tuple: (normality, homoscedasticity, parametric)
               - normality: True if all groups are normally distributed
               - homoscedasticity: True if variances are equal
               - parametric: True if both conditions are satisfied
    """

    normality = True
    homoscedasticity = True
    parametric = True

    # --- Normality Test ---
    p_values = [normaltest(values)[1] for values in groups]

    if any(np.isnan(p) for p in p_values):
        normality = False
        print("Issue with normality check (NaN values detected).")
    
    if any(p < 0.05 for p in p_values):
        normality = False
        print("Data is not normally distributed.")

    # --- Homoscedasticity Test ---
    stat, p_value = levene(*groups)

    if np.isnan(p_value):
        homoscedasticity = False
        print("Issue with homoscedasticity check (NaN value detected).")
    elif p_value < 0.05:
        homoscedasticity = False
        print("Data does not have equal variances.")

    # --- Final Decision ---
    if not (normality and homoscedasticity):
        parametric = False
        print("Non-parametric tests should be used instead.")
    else:
        print("Parametric assumptions met (normality & equal variances).")

    return normality, homoscedasticity, parametric

def calc_stats(title, *groups, debug=True):
    """
    Calculates statistical tests depending on parametric assumptions and number of groups.
    
    Parameters:
        title (str): Label for the dataset or plot.
        *groups: Two or more array-like groups of numeric values.
        debug (bool): Whether to print intermediate results.
    
    Returns:
        dict: Dictionary containing test results and key statistics.
    """
    # Check minimum number of groups
    if len(groups) < 2:
        raise ValueError("At least two groups are required for statistical comparison.")
    
    # --- Test assumptions ---
    normality, homoscedasticity, parametric = test_parametric_conditions(*groups)
    n_groups = len(groups)

    test1, test2 = '', ''
    F_stat, p_val = np.nan, np.nan
    final_stats = None

    # --- PARAMETRIC CASE ---
    if parametric:
        print("Parametric test")
        
        if n_groups == 2:
            test1 = 't-test (independent samples)'
            F_stat, p_val = ttest_ind(groups[0], groups[1])
        
        else:
            test1 = 'One-way ANOVA'
            F_stat, p_val = f_oneway(*groups)

            if p_val < 0.05:
                test2='Tukey test'
                data = np.concatenate(groups)
                labels = np.concatenate([[f"label{i+1}"] * len(g) for i, g in enumerate(groups)])
                Tukey_result = tukey_hsd(data, labels)  
                final_stats = Tukey_result.pvalue
    
    # --- NON-PARAMETRIC CASE ---
    else:
        print("Non-parametric test")
        
        if n_groups == 2:
            test1 = 'Mann-Whitney U test'
            F_stat, p_val = mannwhitneyu(groups[0], groups[1])
        
        else:
            test1 = 'Kruskal-Wallis test'
            F_stat, p_val = kruskal(*groups)

            if p_val < 0.05:
                test2 = "Dunn's post-hoc test (Bonferroni-adjusted)"
                data = {"Value": np.concatenate(groups),
                        "Group": np.concatenate([[f"label{i+1}"] * len(g) for i, g in enumerate(groups)])}
                df = pd.DataFrame(data)
                final_stats = sp.posthoc_dunn(df, val_col="Value", group_col="Group", p_adjust='bonferroni')

    # --- Compile results ---
    stats = {
        'Barplot': title,
        'Normality': normality,
        'Homoscedasticity': homoscedasticity,
        'Parametric': parametric,
        'Test': test1,
        'F_stat': F_stat,
        'p_val': p_val,
        'Supplementary test': test2 if test2 else None,
        'final_stats': final_stats
    }

    if debug:
        print("\n=== Statistical Summary ===")
        for k, v in stats.items():
            print(f"{k}: {v}")
        print("===========================\n")

    return stats

def plot_stats(ax, n_groups, stats, y_pos1=1, y_pos2=1):
    
    if n_groups==2:
        y_pos_m = 0
        significance = 'ns'
        p_value = stats['p_val']
        if (np.isnan(p_value)):
            significance = 'ns'  # Default is "not significant"
        elif (p_value>0.05):
            significance = 'ns'  # Default is "not significant"
        elif (p_value < 0.001):
            significance = '***'
        elif (p_value < 0.01):
            significance = '**'
        elif (p_value < 0.05):
            significance = '*'

        if significance!='ns':

            # Get dynamic bar height = 1/8 of axis range
            ymin, ymax = ax.get_ylim()
            print("ymin, ymax :", ymin, ymax)
            bracket_height = (ymax - ymin) / 8.0

            # Get the y positions for both bars being compared
            #y_pos1 = means[0] + sems[0]
            #y_pos2 = means[1] + sems[1]
            y_pos = max(y_pos1, y_pos2) + ((ymax - ymin)/2.5) # Place the significance line above the highest bar
            if y_pos==y_pos_m:
                y_pos += (ymax - ymin)/2.5
            y_pos_m=y_pos
         
            # Draw a line between the bars
            ax.plot([0, 1], [y_pos, y_pos], color='black', lw=0.9)
            ax.plot([0, 0], [y_pos - bracket_height, y_pos], color='black', lw=0.9)
            ax.plot([1, 1], [y_pos - bracket_height, y_pos], color='black', lw=0.9)
         
            # Annotate the significance above the line
            ax.text((0 + 1) / 2, y_pos, f"{significance}", ha='center', va='bottom', fontsize=15)
    
    
    if n_groups==3 :
        if stats['p_val'] < 0.05: #there are differences between groups
            y_pos_m = 0
            for j1 in range(n_groups):
                for j2 in range(n_groups):
                    if j1 < j2:  # Avoid duplicate comparisons (i.e., comparing the same time to itself)
                        significance = 'ns'
                        p_value = stats['final_stats'].iloc[j1, j2]
                        if (np.isnan(p_value).all()):
                            significance = 'ns'  # Default is "not significant"
                        elif (p_value>0.05):
                            significance = 'ns'  # Default is "not significant"
                        elif (p_value < 0.001):
                            significance = '***'
                        elif (p_value < 0.01):
                            significance = '**'
                        elif (p_value < 0.05):

                            significance = '*'

                        if significance!='ns':
                            # Get the y positions for both bars being compared
                            #y_pos1 = means[j1] + sems[j1]
                            #y_pos2 = means[j2] + sems[j2]
                            y_pos = max(y_pos1, y_pos2) + 0.2 # Place the significance line above the highest bar
                            if y_pos==y_pos_m:
                                y_pos +=1
                            y_pos_m=y_pos
                        
                            # Get dynamic bar height = 1/6 of axis range
                            ymin, ymax = ax.get_ylim()
                            print("ymin, ymax :", ymin, ymax)
                            bracket_height = (ymax - ymin) / 6.0

                            # Draw a line between the bars
                            ax.plot([j1, j2], [y_pos, y_pos], color='black', lw=0.9)
                            ax.plot([j1, j1], [y_pos - bracket_height, y_pos], color='black', lw=0.9)
                            ax.plot([j2, j2], [y_pos - bracket_height, y_pos], color='black', lw=0.9)
                                            
                            # Annotate the significance above the line
                            ax.text((j1 + j2) / 2, y_pos + 0.02, f"{significance}", ha='center', va='bottom', fontsize=15)

if __name__=='__main__':
    print("statistics")
    print("SHOW how tests work here")
