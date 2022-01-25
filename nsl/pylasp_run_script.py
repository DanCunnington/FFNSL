#ilasp_script
import time
start_time = time.time()
ilasp.cdilp.initialise()
solve_result = ilasp.cdilp.solve()

# Stop after iterations take longer than 1 min, whole task is taking 15 mins or a score of 500 is reached, whichever comes first
stopping_bound = 500
iteration_time_cut_off = 5 * 60
time_cut_off = 15 * 60

conflict_analysis_strategy = {
  'positive-strategy': 'all-ufs',
  'negative-strategy': 'single-as',
  'brave-strategy':    'all-ufs',
  'cautious-strategy': 'single-as-pair'
}

c_egs = None
best_score = -1
expected_score = 0
best_solve = solve_result

if solve_result:
  c_egs = ilasp.find_all_counterexamples(solve_result)
  expected_score = solve_result['expected_score']
  true_score = solve_result['expected_score']
  for ce_i in c_egs:
    true_score += ilasp.get_example(ce_i)['penalty']
  best_score = true_score
  best_solve = solve_result

prev_iteration_time = 0
current_time = time.time()

while solve_result is not None and c_egs and (best_score - expected_score) > stopping_bound and \
  prev_iteration_time < iteration_time_cut_off and (current_time - start_time) < time_cut_off:
  it_start_time = time.time()
  ce = ilasp.get_example(c_egs[0])
  constraint = ilasp.cdilp.analyse_conflict(solve_result['hypothesis'], ce['id'], conflict_analysis_strategy)

  # An example with recorded penalty of 0 is in reality an example with an
  # infinite penalty, meaning that it must be covered. Constraint propagation is,
  # therefore, unnecessary.
  if not ce['penalty'] == 0:
    prop_egs = []
    if ce['type'] == 'positive':
      prop_egs = ilasp.cdilp.propagate_constraint(constraint, c_egs, {'select-examples': ['positive'], 'strategy': 'cdpi-implies-constraint'})
    elif ce['type'] == 'negative':
      prop_egs = ilasp.cdilp.propagate_constraint(constraint, c_egs, {'select-examples': ['negative'], 'strategy': 'neg-constraint-implies-cdpi'})
    elif ce['type'] == 'brave-order':
      prop_egs = ilasp.cdilp.propagate_constraint(constraint, c_egs, {'select-examples': ['brave-order'],    'strategy': 'cdoe-implies-constraint'})
    else:
      prop_egs = [ce['id']]

    ilasp.cdilp.add_coverage_constraint(constraint, prop_egs)

  else:
    ilasp.cdilp.add_coverage_constraint(constraint, [ce['id']])

  solve_result = ilasp.cdilp.solve()

  if solve_result is not None:
    c_egs = ilasp.find_all_counterexamples(solve_result)

    expected_score = solve_result['expected_score']
    true_score = solve_result['expected_score']
    for ce_i in c_egs:
      true_score += ilasp.get_example(ce_i)['penalty']
    if best_score == -1 or best_score > true_score:
      best_score = true_score
      best_solve = solve_result

  it_end_time = time.time()
  prev_iteration_time = it_end_time - it_start_time
  current_time = it_end_time
if best_solve:
  print ilasp.hypothesis_to_string(best_solve['hypothesis'])
else:
  print 'UNSATISFIABLE'

ilasp.stats.print_timings()

#end.
