% Task 6: Prolog as a Plan Verifier

% Facts describing the warehouse
connected(a,b).
connected(b,a).
connected(b,c).
connected(c,b).

% Rules
can_move(X,Y) :-
    connected(X,Y).

% Task 7: Using Prolog to Check a Proposed Plan
valid_move(X,Y) :-
    connected(X,Y).

% Task 8: Connect Prolog to Logical Reasoning
wet_road.

slippery :-
    wet_road.

reduce_speed :-
    slippery.
