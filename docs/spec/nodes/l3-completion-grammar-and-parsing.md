# Completion grammar and parsing

A generated completion consists of one or more nonempty reasoning walks
separated by restart markers, followed by the end-reasoning marker, the
begin-answer marker, a nonempty node-label sequence, the end-answer marker, and
the termination symbol. The complete reasoning segment contains at least the
declared positive `minimum_reason_tokens` tokens, counting node labels and
restart markers. A restart marker must occur between walks, and the termination
symbol occurs exactly once at the end.

Parsing validates the minimum-reasoning declaration, decodes node tokens using
the complete vocabulary, and returns the raw reasoning tokens, separated
reasoning walks, and answer path. A reasoning segment shorter than the declared
minimum, a structural token in a walk or answer, a missing or misplaced
delimiter, an empty required segment, or invalid termination produces a format
failure. Parsing does not consult the active graph or judge transition legality,
endpoint correctness, or optimality.
