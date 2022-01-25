background_knowledge = '''
% Suits
suit(h).
suit(s).
suit(d).
suit(c).

% Ranks
rank(a).
rank(2).
rank(3).
rank(4).
rank(5).
rank(6).
rank(7).
rank(8).
rank(9).
rank(10).
rank(j).
rank(q).
rank(k).

% Rank Value
rank_value(2, 2).
rank_value(3, 3).
rank_value(4, 4).
rank_value(5, 5).
rank_value(6, 6).
rank_value(7, 7).
rank_value(8, 8).
rank_value(9, 9).
rank_value(10, 10).
rank_value(j, 11).
rank_value(q, 12).
rank_value(k, 13).
rank_value(a, 14).

% 4 Players
player(1..4).

% Definition of higher rank
rank_higher(P1, P2) :- card(P1, R1, _), card(P2, R2, _), rank_value(R1, V1), rank_value(R2, V2), V1 > V2.

% Link player's card to suit
suit(P1, S) :- card(P1, _, S).
'''


mode_declarations = '''
P(X) :- Q(X), identity(P, Q).
P(X) :- player(X), not Q(X), inverse(P, Q).
#modem(2, inverse(target/1, invented/1)).
#modem(2, identity(target/1, invented/1)).
#predicate(target, winner/1).
#predicate(invented, p1/1).

#constant(player, 1).
#constant(player, 2).
#constant(player, 3).
#constant(player, 4).
#modeh(p1(var(player))).
#modeb(1, var(suit) != var(suit)).
#modeb(1, suit(var(player), var(suit)), (positive)).
#modeb(1, suit(const(player), var(suit)), (positive)).
#modeb(1, rank_higher(var(player), var(player)), (positive)).
'''