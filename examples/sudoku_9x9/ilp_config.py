background_knowledge = '''
value("1, 1", X) :- value(1, 1, X).
value("1, 2", X) :- value(1, 2, X).
value("1, 3", X) :- value(1, 3, X).
value("1, 4", X) :- value(1, 4, X).
value("1, 5", X) :- value(1, 5, X).
value("1, 6", X) :- value(1, 6, X).
value("1, 7", X) :- value(1, 7, X).
value("1, 8", X) :- value(1, 8, X).
value("1, 9", X) :- value(1, 9, X).

value("2, 1", X) :- value(2, 1, X).
value("2, 2", X) :- value(2, 2, X).
value("2, 3", X) :- value(2, 3, X).
value("2, 4", X) :- value(2, 4, X).
value("2, 5", X) :- value(2, 5, X).
value("2, 6", X) :- value(2, 6, X).
value("2, 7", X) :- value(2, 7, X).
value("2, 8", X) :- value(2, 8, X).
value("2, 9", X) :- value(2, 9, X).

value("3, 1", X) :- value(3, 1, X).
value("3, 2", X) :- value(3, 2, X).
value("3, 3", X) :- value(3, 3, X).
value("3, 4", X) :- value(3, 4, X).
value("3, 5", X) :- value(3, 5, X).
value("3, 6", X) :- value(3, 6, X).
value("3, 7", X) :- value(3, 7, X).
value("3, 8", X) :- value(3, 8, X).
value("3, 9", X) :- value(3, 9, X).

value("4, 1", X) :- value(4, 1, X).
value("4, 2", X) :- value(4, 2, X).
value("4, 3", X) :- value(4, 3, X).
value("4, 4", X) :- value(4, 4, X).
value("4, 5", X) :- value(4, 5, X).
value("4, 6", X) :- value(4, 6, X).
value("4, 7", X) :- value(4, 7, X).
value("4, 8", X) :- value(4, 8, X).
value("4, 9", X) :- value(4, 9, X).

value("5, 1", X) :- value(5, 1, X).
value("5, 2", X) :- value(5, 2, X).
value("5, 3", X) :- value(5, 3, X).
value("5, 4", X) :- value(5, 4, X).
value("5, 5", X) :- value(5, 5, X).
value("5, 6", X) :- value(5, 6, X).
value("5, 7", X) :- value(5, 7, X).
value("5, 8", X) :- value(5, 8, X).
value("5, 9", X) :- value(5, 9, X).

value("6, 1", X) :- value(6, 1, X).
value("6, 2", X) :- value(6, 2, X).
value("6, 3", X) :- value(6, 3, X).
value("6, 4", X) :- value(6, 4, X).
value("6, 5", X) :- value(6, 5, X).
value("6, 6", X) :- value(6, 6, X).
value("6, 7", X) :- value(6, 7, X).
value("6, 8", X) :- value(6, 8, X).
value("6, 9", X) :- value(6, 9, X).

value("7, 1", X) :- value(7, 1, X).
value("7, 2", X) :- value(7, 2, X).
value("7, 3", X) :- value(7, 3, X).
value("7, 4", X) :- value(7, 4, X).
value("7, 5", X) :- value(7, 5, X).
value("7, 6", X) :- value(7, 6, X).
value("7, 7", X) :- value(7, 7, X).
value("7, 8", X) :- value(7, 8, X).
value("7, 9", X) :- value(7, 9, X).

value("8, 1", X) :- value(8, 1, X).
value("8, 2", X) :- value(8, 2, X).
value("8, 3", X) :- value(8, 3, X).
value("8, 4", X) :- value(8, 4, X).
value("8, 5", X) :- value(8, 5, X).
value("8, 6", X) :- value(8, 6, X).
value("8, 7", X) :- value(8, 7, X).
value("8, 8", X) :- value(8, 8, X).
value("8, 9", X) :- value(8, 9, X).

value("9, 1", X) :- value(9, 1, X).
value("9, 2", X) :- value(9, 2, X).
value("9, 3", X) :- value(9, 3, X).
value("9, 4", X) :- value(9, 4, X).
value("9, 5", X) :- value(9, 5, X).
value("9, 6", X) :- value(9, 6, X).
value("9, 7", X) :- value(9, 7, X).
value("9, 8", X) :- value(9, 8, X).
value("9, 9", X) :- value(9, 9, X).

% Columns
col("1, 1", 1).
col("1, 2", 2).
col("1, 3", 3).
col("1, 4", 4).
col("1, 5", 5).
col("1, 6", 6).
col("1, 7", 7).
col("1, 8", 8).
col("1, 9", 9).

col("2, 1", 1).
col("2, 2", 2).
col("2, 3", 3).
col("2, 4", 4).
col("2, 5", 5).
col("2, 6", 6).
col("2, 7", 7).
col("2, 8", 8).
col("2, 9", 9).

col("3, 1", 1).
col("3, 2", 2).
col("3, 3", 3).
col("3, 4", 4).
col("3, 5", 5).
col("3, 6", 6).
col("3, 7", 7).
col("3, 8", 8).
col("3, 9", 9).

col("4, 1", 1).
col("4, 2", 2).
col("4, 3", 3).
col("4, 4", 4).
col("4, 5", 5).
col("4, 6", 6).
col("4, 7", 7).
col("4, 8", 8).
col("4, 9", 9).

col("5, 1", 1).
col("5, 2", 2).
col("5, 3", 3).
col("5, 4", 4).
col("5, 5", 5).
col("5, 6", 6).
col("5, 7", 7).
col("5, 8", 8).
col("5, 9", 9).

col("6, 1", 1).
col("6, 2", 2).
col("6, 3", 3).
col("6, 4", 4).
col("6, 5", 5).
col("6, 6", 6).
col("6, 7", 7).
col("6, 8", 8).
col("6, 9", 9).

col("7, 1", 1).
col("7, 2", 2).
col("7, 3", 3).
col("7, 4", 4).
col("7, 5", 5).
col("7, 6", 6).
col("7, 7", 7).
col("7, 8", 8).
col("7, 9", 9).

col("8, 1", 1).
col("8, 2", 2).
col("8, 3", 3).
col("8, 4", 4).
col("8, 5", 5).
col("8, 6", 6).
col("8, 7", 7).
col("8, 8", 8).
col("8, 9", 9).

col("9, 1", 1).
col("9, 2", 2).
col("9, 3", 3).
col("9, 4", 4).
col("9, 5", 5).
col("9, 6", 6).
col("9, 7", 7).
col("9, 8", 8).
col("9, 9", 9).

% Rows
row("1, 1", 1).
row("1, 2", 1).
row("1, 3", 1).
row("1, 4", 1).
row("1, 5", 1).
row("1, 6", 1).
row("1, 7", 1).
row("1, 8", 1).
row("1, 9", 1).

row("2, 1", 2).
row("2, 2", 2).
row("2, 3", 2).
row("2, 4", 2).
row("2, 5", 2).
row("2, 6", 2).
row("2, 7", 2).
row("2, 8", 2).
row("2, 9", 2).

row("3, 1", 3).
row("3, 2", 3).
row("3, 3", 3).
row("3, 4", 3).
row("3, 5", 3).
row("3, 6", 3).
row("3, 7", 3).
row("3, 8", 3).
row("3, 9", 3).

row("4, 1", 4).
row("4, 2", 4).
row("4, 3", 4).
row("4, 4", 4).
row("4, 5", 4).
row("4, 6", 4).
row("4, 7", 4).
row("4, 8", 4).
row("4, 9", 4).

row("5, 1", 5).
row("5, 2", 5).
row("5, 3", 5).
row("5, 4", 5).
row("5, 5", 5).
row("5, 6", 5).
row("5, 7", 5).
row("5, 8", 5).
row("5, 9", 5).

row("6, 1", 6).
row("6, 2", 6).
row("6, 3", 6).
row("6, 4", 6).
row("6, 5", 6).
row("6, 6", 6).
row("6, 7", 6).
row("6, 8", 6).
row("6, 9", 6).

row("7, 1", 7).
row("7, 2", 7).
row("7, 3", 7).
row("7, 4", 7).
row("7, 5", 7).
row("7, 6", 7).
row("7, 7", 7).
row("7, 8", 7).
row("7, 9", 7).

row("8, 1", 8).
row("8, 2", 8).
row("8, 3", 8).
row("8, 4", 8).
row("8, 5", 8).
row("8, 6", 8).
row("8, 7", 8).
row("8, 8", 8).
row("8, 9", 8).

row("9, 1", 9).
row("9, 2", 9).
row("9, 3", 9).
row("9, 4", 9).
row("9, 5", 9).
row("9, 6", 9).
row("9, 7", 9).
row("9, 8", 9).
row("9, 9", 9).

% Blocks
block("1, 1", 1).
block("1, 2", 1).
block("1, 3", 1).
block("2, 1", 1).
block("2, 2", 1).
block("2, 3", 1).
block("3, 1", 1).
block("3, 2", 1).
block("3, 3", 1).

block("1, 4", 2).
block("1, 5", 2).
block("1, 6", 2).
block("2, 4", 2).
block("2, 5", 2).
block("2, 6", 2).
block("3, 4", 2).
block("3, 5", 2).
block("3, 6", 2).

block("1, 7", 3).
block("1, 8", 3).
block("1, 9", 3).
block("2, 7", 3).
block("2, 8", 3).
block("2, 9", 3).
block("3, 7", 3).
block("3, 8", 3).
block("3, 9", 3).

block("4, 1", 4).
block("4, 2", 4).
block("4, 3", 4).
block("5, 1", 4).
block("5, 2", 4).
block("5, 3", 4).
block("6, 1", 4).
block("6, 2", 4).
block("6, 3", 4).

block("4, 4", 5).
block("4, 5", 5).
block("4, 6", 5).
block("5, 4", 5).
block("5, 5", 5).
block("5, 6", 5).
block("6, 4", 5).
block("6, 5", 5).
block("6, 6", 5).

block("4, 7", 6).
block("4, 8", 6).
block("4, 9", 6).
block("5, 7", 6).
block("5, 8", 6).
block("5, 9", 6).
block("6, 7", 6).
block("6, 8", 6).
block("6, 9", 6).

block("7, 1", 7).
block("7, 2", 7).
block("7, 3", 7).
block("8, 1", 7).
block("8, 2", 7).
block("8, 3", 7).
block("9, 1", 7).
block("9, 2", 7).
block("9, 3", 7).

block("7, 4", 8).
block("7, 5", 8).
block("7, 6", 8).
block("8, 4", 8).
block("8, 5", 8).
block("8, 6", 8).
block("9, 4", 8).
block("9, 5", 8).
block("9, 6", 8).

block("7, 7", 9).
block("7, 8", 9).
block("7, 9", 9).
block("8, 7", 9).
block("8, 8", 9).
block("8, 9", 9).
block("9, 7", 9).
block("9, 8", 9).
block("9, 9", 9).

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
num(1..9).
row(1..9).
col(1..9).
block(1..9).
cell(C) :- value(C, _).

#bias("penalty(1, head).").
#bias("penalty(1, body(X)) :- in_body(X).").
'''