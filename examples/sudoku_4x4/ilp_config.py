background_knowledge = '''
value("1, 1", X) :- value(1, 1, X).
value("1, 2", X) :- value(1, 2, X).
value("1, 3", X) :- value(1, 3, X).
value("1, 4", X) :- value(1, 4, X).

value("2, 1", X) :- value(2, 1, X).
value("2, 2", X) :- value(2, 2, X).
value("2, 3", X) :- value(2, 3, X).
value("2, 4", X) :- value(2, 4, X).

value("3, 1", X) :- value(3, 1, X).
value("3, 2", X) :- value(3, 2, X).
value("3, 3", X) :- value(3, 3, X).
value("3, 4", X) :- value(3, 4, X).

value("4, 1", X) :- value(4, 1, X).
value("4, 2", X) :- value(4, 2, X).
value("4, 3", X) :- value(4, 3, X).
value("4, 4", X) :- value(4, 4, X).

% Columns
col("1, 1", 1).
col("1, 2", 2).
col("1, 3", 3).
col("1, 4", 4).

col("2, 1", 1).
col("2, 2", 2).
col("2, 3", 3).
col("2, 4", 4).

col("3, 1", 1).
col("3, 2", 2).
col("3, 3", 3).
col("3, 4", 4).

col("4, 1", 1).
col("4, 2", 2).
col("4, 3", 3).
col("4, 4", 4).

% Rows
row("1, 1", 1).
row("1, 2", 1).
row("1, 3", 1).
row("1, 4", 1).

row("2, 1", 2).
row("2, 2", 2).
row("2, 3", 2).
row("2, 4", 2).

row("3, 1", 3).
row("3, 2", 3).
row("3, 3", 3).
row("3, 4", 3).

row("4, 1", 4).
row("4, 2", 4).
row("4, 3", 4).
row("4, 4", 4).


% Blocks
block("1, 1", 1).
block("1, 2", 1).
block("2, 1", 1).
block("2, 2", 1).

block("1, 3", 2).
block("1, 4", 2).
block("2, 3", 2).
block("2, 4", 2).

block("3, 1", 3).
block("3, 2", 3).
block("4, 1", 3).
block("4, 2", 3).

block("3, 3", 4).
block("3, 4", 4).
block("4, 3", 4).
block("4, 4", 4).

'''

mode_declarations = '''
#modeh(invalid).
#modeb(value(var(cell), var(num))).
#modeb(row(var(cell), var(row))).
#modeb(not row(var(cell), var(row))).
#modeb(col(var(cell), var(col))).
#modeb(not col(var(cell), var(col))).
#modeb(block(var(cell), var(block))).
#modeb(not block(var(cell), var(block))).
#modeb(neq(var(cell), var(cell))).
neq(X, Y) :- cell(X), cell(Y), X != Y.

#maxv(4).
num(1..4).
row(1..4).
col(1..4).
block(1..4).
cell(C) :- value(C, _).

#bias("penalty(1, head).").
#bias("penalty(1, body(X)) :- in_body(X).").
'''

reduced_background_knowledge = '''
div_same1(X, Y, C) :- (X - 1) / C = (Y - 1) / C, idx1(X), idx1(Y), X < Y, quotient(C).
div_same2(X, Y, C) :- (X - 1) / C = (Y - 1) / C, idx2(X), idx2(Y), X < Y, quotient(C).

quotient(1..3).
idx1(1..4).
idx2(1..4).
num(1..4).
'''

reduced_background_knowledge_for_problog = '''
quotient(1).
quotient(2).
quotient(3).
idx1(1).
idx1(2).
idx1(3).
idx1(4).
idx2(1).
idx2(2).
idx2(3).
idx2(4).
num(1).
num(2).
num(3).
num(4).

math_op(A,C,R) :- R is (A-1)//C.
div_same1(X, Y, C) :- idx1(X), idx1(Y), X < Y, quotient(C),  math_op(X,C,R), math_op(Y,C,R).
div_same2(X, Y, C) :- idx2(X), idx2(Y), X < Y, quotient(C),  math_op(X,C,R), math_op(Y,C,R).

'''

reduced_mode_declarations = '''
#modeh(invalid).
#modeb(value(var(idx1), var(idx2), var(num))).
#modeb(div_same1(var(idx1), var(idx1), const(quotient))).
#modeb(div_same2(var(idx2), var(idx2), const(quotient))).

#maxv(5).

#bias("penalty(1, head).").
#bias("penalty(1, body(X)) :- in_body(X).").
#ground_without_replacement.
'''