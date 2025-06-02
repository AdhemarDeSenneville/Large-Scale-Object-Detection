from math import factorial, exp


def get_score(bd_confidence, tank_confidence, pile_confidence, methode):

    tank_count_prob = prob_distribution(tank_confidence)
    pile_count_prob = prob_distribution(pile_confidence)
    
    if methode == 'acontrario_poisson':
        probability_distribution = acontrario_poisson
    if methode == 'acontrario_histogram':
        probability_distribution = acontrario_histogram
    if methode == 'proba_poisson':
        probability_distribution = proba_poisson
    if methode == 'proba_histogram':
        probability_distribution = proba_histogram
    if methode == 'mixed' or methode == 'baseline':
        probability_distribution = mixed

    total = 0.0
    for tank_count in range(len(tank_count_prob)):
        for pile_count in range(len(pile_count_prob)):
            total += tank_count_prob[tank_count] * pile_count_prob[pile_count] * probability_distribution(tank_count, pile_count)

    if methode == 'baseline':
        total = bd_confidence
    else:
        total *= bd_confidence

    return total

def detection_function(biodigester, detections, methode):
    
    id_tank = 2
    id_pile = 3

    tank_found = []
    pile_found = []
    detections_inside = [biodigester]

    for object in detections:
        # Check if the center is inside the current biodigester bbox
        if is_center_inside(biodigester['bbox'], object['bbox']):
            if object['category_id'] == id_tank:
                tank_found.append(object['score'])
                detections_inside.append(object)
            if object['category_id'] == id_pile:
                pile_found.append(object['score'])
                detections_inside.append(object)

    tank_count_prob = prob_distribution(tank_found)
    pile_count_prob = prob_distribution(pile_found)

    if methode == 'acontrario_poisson':
        probability_distribution = acontrario_poisson
    if methode == 'acontrario_histogram':
        probability_distribution = acontrario_histogram
    if methode == 'proba_poisson':
        probability_distribution = proba_poisson
    if methode == 'proba_histogram':
        probability_distribution = proba_histogram
    if methode == 'mixed' or methode == 'baseline':
        probability_distribution = mixed

    total = 0.0
    for tank_count in range(len(tank_count_prob)):
        for pile_count in range(len(pile_count_prob)):
            total += tank_count_prob[tank_count] * pile_count_prob[pile_count] * probability_distribution(tank_count, pile_count)

    if methode == 'baseline':
        total = biodigester['score']
    else:
        total *= biodigester['score']    

    return detections_inside, total


def prob_distribution(prob_list):
    """
    Given a list of n probabilities, returns a list of length n+1,
    where the k-th element is the probability of exactly k successes.
    """
    n = len(prob_list)
    dp = [0.0] * (n + 1)
    dp[0] = 1.0  # Initially, the probability of 0 successes (with 0 trials) is 1
    
    for p in prob_list:
        # Update from right to left
        for k in range(n, 0, -1):
            dp[k] = dp[k] * (1 - p) + dp[k - 1] * p
        dp[0] = dp[0] * (1 - p)
    
    return dp


####################################################
# Probabilistic identification of false alarms
####################################################


def proba_poisson(k1, k2, lam1 = 2.99, lam2 = 4.8762599, lam12 = -0.01419): # DONE
    """Compute the Bivariate Poisson probability p(k1, k2)."""
    pmf_sum = 0.0
    limit = min(k1, k2)
    for j in range(limit + 1):
        pmf_sum += (lam1**(k1-j) / factorial(k1-j)) \
                 * (lam2**(k2-j) / factorial(k2-j)) \
                 * (lam12**j      / factorial(j))
    return exp(-(lam1 + lam2 + lam12)) * pmf_sum


def proba_histogram(k1, k2): # DONE
    """Compute the histogram probability p(k1, k2)."""
    p1_histogram = [0, 0.06896552, 0.31417625, 0.36015326, 0.14942529, 0.06896552, 0.01532567, 0.01915709, 0.00383142]
    p2_histogram = [0.01532567, 0.06130268, 0.10344828, 0.16091954, 0.16475096, 0.12643678, 0.11494253, 0.08812261, 
                    0.07279693, 0.04597701, 0.01915709, 0.00766284, 0.01149425, 0.00383142, 0.00383142]

    if k1 < len(p1_histogram):
        p1 = p1_histogram[k1]
    else:
        p1 = 0

    if k2 < len(p2_histogram):
        p2 = p2_histogram[k2]
    else:
        p2 = 0

    return p1 * p2


####################################################
# A-contrario identification of false alarms 
####################################################


def acontrario_poisson(k1, k2, lam1 = 0.3047, lam2 = 0.35079, lam12 = 0.20643):
    """Compute the Bivariate Poisson probability p(k1, k2)."""
    pmf_sum = 0.0
    limit = min(k1, k2)
    for j in range(limit + 1):
        pmf_sum += (lam1**(k1-j) / factorial(k1-j)) \
                 * (lam2**(k2-j) / factorial(k2-j)) \
                 * (lam12**j      / factorial(j))
    return 1 - exp(-(lam1 + lam2 + lam12)) * pmf_sum


def acontrario_histogram(k1, k2):
    """Compute the histogram probability p(k1, k2)."""
    p1_histogram = [0.88376, 0.08587, 0.01811, 0.00611, 0.00287, 0.00137, 0.00072, 0.00048, 0.00035, 0.00021, 0.00010, 0.00004, 0.00001]
    p2_histogram = [0.93005, 0.05367, 0.01218, 0.00303, 0.00075, 0.00022, 0.00007, 0.00002, 0.00001]

    if k1 < len(p1_histogram):
        p1 = p1_histogram[k1]
    else:
        p1 = 0

    if k2 < len(p2_histogram):
        p2 = p2_histogram[k2]
    else:
        p2 = 0

    return 1 - p1 * p2


####################################################
# Mixed identification of false alarms 
####################################################


def mixed(k1, k2):

    p_methane = proba_histogram(k1, k2)
    p_anomaly = acontrario_histogram(k1, k2)

    return p_methane/(1-p_anomaly + 1e-6)


####################################################


def is_center_inside(bbox_all, bbox_check):
    """ Check if the center (x, y) is inside the bbox_all (x1, y1, x2, y2). """

    cx = bbox_check[0] + bbox_check[2] / 2
    cy = bbox_check[1] + bbox_check[3] / 2

    x, y, w, h = bbox_all
    return x <= cx <= x + w and y <= cy <= y + h