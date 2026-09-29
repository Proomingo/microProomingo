# micro_proomingo.py — Premier micro-perceptron autonome apprenti de la porte logique ET (AND)

import random  # Importation du module random pour initialiser les poids de maniere aleatoire au depart

# --- 1. DEFINITION DE LA DONNEE D'ENTRAINEMENT (La table logique AND) ---
# Chaque sous-liste contient [Entree_1, Entree_2]
donnees_entree = [[0, 0], [0, 1], [1, 0], [1, 1]]  # Les 4 combinaisons possibles de deux bits
# La reponse exacte attendue pour chaque combinaison (Seul [1, 1] donne 1)
reponses_attendues = [0, 0, 0, 1]  # La cible que MicroProomingo doit apprendre a deviner

# --- 2. INITIALISATION DES PARAMETRES DU PERCEPTRON ---
# Poids assignes a l'entree 1 et l'entree 2, choisis au hasard au depart entre -1.0 et 1.0
poids_1 = random.uniform(-1.0, 1.0)  # Poids aleatoire pour la premiere entree
poids_2 = random.uniform(-1.0, 1.0)  # Poids aleatoire pour la deuxieme entree
# Biais pour regler le seuil d'excitabilite du neurone
biais = random.uniform(-1.0, 1.0)  # Biais aleatoire initial
# Taux d'apprentissage (cadence d'ajustement des reglages)
taux_apprentissage = 0.1  # Petit pas pour eviter des corrections trop violentes

# --- 3. FONCTION D'ACTIVATION (FONCTION SEUIL) ---
def activation(somme):  # Prend la note globale calculée par le neurone
    if somme >= 0:  # Si la note depasse ou egale le seuil de 0
        return 1  # Le neurone s'active et renvoie 1
    else:  # Si la note est negative
        return 0  # Le neurone reste silencieux et renvoie 0

# --- 4. BOUCLE D'ENTRAINEMENT (APPRENTISSAGE SUR 20 EPOQUES) ---
print("--- DEBUT DE L'ENTRAINEMENT DE MICROPROOMINGO-0 ---")  # Affichage de debut
for epoque in range(20):  # On repete le processus d'apprentissage 20 fois (20 epoques)
    erreur_totale_epoque = 0  # Compteur pour additionner les erreurs commises durant cette epoque
    
    for i in range(len(donnees_entree)):  # Parcours des 4 exemples d'entree un par un
        x1 = donnees_entree[i][0]  # Extraction de la premiere valeur d'entree
        x2 = donnees_entree[i][1]  # Extraction de la deuxieme valeur d'entree
        cible = reponses_attendues[i]  # Extraction de la reponse qu'on attend du neurone
        
        # ETAPE A : Calcul de la somme ponderee (Note globale)
        somme_ponderee = (x1 * poids_1) + (x2 * poids_2) + biais  # Application de la formule (x1*w1 + x2*w2 + b)
        
        # ETAPE B : Prédicton via la fonction d'activation
        prediction = activation(somme_ponderee)  # Conversion de la somme en une decision 0 ou 1
        
        # ETAPE C : Calcul de l'erreur commise
        erreur = cible - prediction  # Ecart entre la vraie valeur et la prediction (ex: 1 - 0 = 1)
        erreur_totale_epoque += abs(erreur)  # Cumul de l'erreur absolue pour le bilan de l'epoque
        
        # ETAPE D : Ajustement des poids et du biais (Règle d'apprentissage Rosenblatt)
        poids_1 = poids_1 + (taux_apprentissage * erreur * x1)  # Correction du poids 1 selon l'erreur et x1
        poids_2 = poids_2 + (taux_apprentissage * erreur * x2)  # Correction du poids 2 selon l'erreur et x2
        biais = biais + (taux_apprentissage * erreur)  # Correction du biais selon l'erreur uniquement
        
    print(f"Epoque {epoque + 1}/20 — Erreurs totales : {erreur_totale_epoque}")  # Affichage du bilan de l'epoque
    if erreur_totale_epoque == 0:  # Si MicroProomingo ne fait plus aucune erreur sur les 4 cas
        print("MicroProomingo a parfaitement appris la logique !")  # Message de succes
        break  # Interruption anticipee de la boucle d'entrainement

# --- 5. TEST ET VERIFICATION DES RESULTATS FINAUX ---
print("\n--- TEST FINAL DES DECISIONS DE MICROPROOMINGO-0 ---")  # Titre de la section test
for i in range(len(donnees_entree)):  # Parcours des cas de test
    x1 = donnees_entree[i][0]  # Entree 1
    x2 = donnees_entree[i][1]  # Entree 2
    somme = (x1 * poids_1) + (x2 * poids_2) + biais  # Calcul final avec les poids optimises
    resultat = activation(somme)  # Prise de decision finale
    print(f"Entrees: [{x1}, {x2}] -> Decision de Proomingo: {resultat} (Attendu: {reponses_attendues[i]})")  # Bilan