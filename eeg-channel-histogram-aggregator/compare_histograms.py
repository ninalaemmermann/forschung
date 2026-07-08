"""
Berechnet JS-Divergenz zwischen allen Seizure-Type Histogrammen
"""
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon
import os


def load_histogram(seizure_type):
    """Lädt 27D-Histogram aus PKL-Datei"""
    base_dir = '/home/data/ninalaemmermann/forschung'
    filepath = os.path.join(base_dir, f'histogram_27d_{seizure_type.lower()}.pkl')
    
    print(f"Lade {seizure_type}...")
    with open(filepath, 'rb') as f:
        histogram = pickle.load(f)
    
    return histogram


def sparse_to_probability(sparse_hist, all_bins):
    """
    Konvertiert Sparse-Histogram zu Wahrscheinlichkeitsvektor
    
    Parameters:
    -----------
    sparse_hist : dict
        Sparse histogram {bin_tuple: count}
    all_bins : set
        Alle Bins die in irgendeinem Histogram vorkommen
    
    Returns:
    --------
    array : Wahrscheinlichkeitsvektor (normalisiert)
    """
    total = sum(sparse_hist.values())
    
    # Erstelle Vektor für alle Bins
    prob_vector = np.zeros(len(all_bins))
    bin_to_idx = {bin_tuple: idx for idx, bin_tuple in enumerate(sorted(all_bins))}
    
    for bin_tuple, count in sparse_hist.items():
        if bin_tuple in bin_to_idx:
            idx = bin_to_idx[bin_tuple]
            prob_vector[idx] = count / total
    
    # Normalisiere (sollte schon normalisiert sein, aber sicher ist sicher)
    prob_sum = prob_vector.sum()
    if prob_sum > 0:
        prob_vector = prob_vector / prob_sum
    
    return prob_vector


def calculate_js_divergence(hist1, hist2):
    """
    Berechnet JS-Divergenz zwischen zwei Sparse-Histogrammen
    
    Parameters:
    -----------
    hist1, hist2 : dict
        Sparse histograms {bin_tuple: count}
    
    Returns:
    --------
    float : JS-Divergenz
    """
    # Finde alle Bins die in mindestens einem Histogram vorkommen
    all_bins = set(hist1.keys()) | set(hist2.keys())
    
    print(f"  Bins in Hist1: {len(hist1):,}")
    print(f"  Bins in Hist2: {len(hist2):,}")
    print(f"  Union Bins: {len(all_bins):,}")
    print(f"  Überlappung: {len(set(hist1.keys()) & set(hist2.keys())):,}")
    
    # Konvertiere zu Wahrscheinlichkeitsvektoren
    p = sparse_to_probability(hist1, all_bins)
    q = sparse_to_probability(hist2, all_bins)
    
    # Berechne JS-Divergenz
    # jensenshannon gibt sqrt(JS), also quadrieren für echte JS-Divergenz
    js_distance = jensenshannon(p, q)
    js_divergence = js_distance ** 2
    
    return js_divergence


def create_distance_matrix():
    """
    Erstellt Distanz-Matrix aller Histogram-Paare
    """
    print(f"\n{'='*60}")
    print(f"BERECHNE JS-DIVERGENZ DISTANZ-MATRIX")
    print(f"{'='*60}\n")
    
    seizure_types = ['ABSZ', 'CPSZ', 'FNSZ', 'GNSZ']
    
    # Lade alle Histogramme
    histograms = {}
    for st in seizure_types:
        histograms[st] = load_histogram(st)
    
    print(f"\n✓ Alle Histogramme geladen\n")
    
    # Labels für Matrix
    labels = []
    for st in seizure_types:
        labels.append(f'{st}_Seizure')
        labels.append(f'{st}_NonSeiz')
    
    n = len(labels)
    distance_matrix = np.zeros((n, n))
    
    # Berechne JS-Divergenz für alle Paare
    print("Berechne paarweise JS-Divergenzen...\n")
    
    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels):
            if i <= j:  # Nur oberes Dreieck berechnen (symmetrisch)
                st1, type1 = label1.rsplit('_', 1)
                st2, type2 = label2.rsplit('_', 1)
                
                # Wähle richtiges Histogram
                if type1 == 'Seizure':
                    hist1 = histograms[st1]['seizure_hist']
                else:
                    hist1 = histograms[st1]['non_seizure_hist']
                
                if type2 == 'Seizure':
                    hist2 = histograms[st2]['seizure_hist']
                else:
                    hist2 = histograms[st2]['non_seizure_hist']
                
                print(f"{label1} vs {label2}:")
                js_div = calculate_js_divergence(hist1, hist2)
                
                distance_matrix[i, j] = js_div
                distance_matrix[j, i] = js_div  # Symmetrisch
                
                print(f"  → JS-Divergenz: {js_div:.6f}\n")
    
    return distance_matrix, labels


def plot_distance_matrix(distance_matrix, labels):
    """
    Visualisiert Distanz-Matrix als Heatmap
    """
    print("Erstelle Heatmap...\n")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Heatmap
    sns.heatmap(
        distance_matrix,
        annot=True,
        fmt='.4f',
        cmap='YlOrRd',
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        cbar_kws={'label': 'JS-Divergenz'},
        ax=ax
    )
    
    plt.title('JS-Divergenz zwischen Seizure-Type Histogrammen (27D)\n' +
              '50 Bins/Dimension, 100k Samples', 
              fontsize=14, pad=20)
    plt.xlabel('', fontsize=12)
    plt.ylabel('', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Speichern
    output_dir = '/home/data/ninalaemmermann/forschung/eeg-channel-histogram-aggregator/data/output'
    output_path = os.path.join(output_dir, 'js_divergence_matrix_27d.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Gespeichert: {output_path}")
    
    plt.close()


def analyze_results(distance_matrix, labels):
    """
    Analysiert und gibt interessante Ergebnisse aus
    """
    print(f"\n{'='*60}")
    print(f"ANALYSE DER ERGEBNISSE")
    print(f"{'='*60}\n")
    
    n = len(labels)
    
    # Finde kleinste und größte Distanzen (außer Diagonale)
    distances = []
    for i in range(n):
        for j in range(i+1, n):
            distances.append((distance_matrix[i, j], labels[i], labels[j]))
    
    distances.sort()
    
    print("TOP 5 ÄHNLICHSTE PAARE (kleinste JS-Divergenz):")
    for js, l1, l2 in distances[:5]:
        print(f"  {js:.6f}: {l1} <-> {l2}")
    
    print("\nTOP 5 UNTERSCHIEDLICHSTE PAARE (größte JS-Divergenz):")
    for js, l1, l2 in distances[-5:]:
        print(f"  {js:.6f}: {l1} <-> {l2}")
    
    # Intra-Type Separabilität
    print("\nINTRA-TYPE SEPARABILITÄT (Seizure vs NonSeizure):")
    seizure_types = ['ABSZ', 'CPSZ', 'FNSZ', 'GNSZ']
    separabilities = []
    
    for st in seizure_types:
        i = labels.index(f'{st}_Seizure')
        j = labels.index(f'{st}_NonSeiz')
        sep = distance_matrix[i, j]
        separabilities.append((sep, st))
        print(f"  {st}: {sep:.6f}")
    
    separabilities.sort(reverse=True)
    print(f"\nBESTE Detektierbarkeit: {separabilities[0][1]}")
    print(f"SCHLECHTESTE Detektierbarkeit: {separabilities[-1][1]}")
    
    # Inter-Type Ähnlichkeit (nur Seizures)
    print("\nINTER-TYPE ÄHNLICHKEIT (nur Anfälle):")
    seizure_indices = [labels.index(f'{st}_Seizure') for st in seizure_types]
    
    for i, st1 in enumerate(seizure_types):
        for st2 in seizure_types[i+1:]:
            idx1 = labels.index(f'{st1}_Seizure')
            idx2 = labels.index(f'{st2}_Seizure')
            js = distance_matrix[idx1, idx2]
            print(f"  {st1} <-> {st2}: {js:.6f}")


def main():
    """
    Hauptprogramm
    """
    # Berechne Distanz-Matrix
    distance_matrix, labels = create_distance_matrix()
    
    # Visualisiere
    plot_distance_matrix(distance_matrix, labels)
    
    # Analysiere
    analyze_results(distance_matrix, labels)
    
    print(f"\n{'='*60}")
    print(f"✓ FERTIG!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
