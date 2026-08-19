"""Jazz reharmonization: the one-to-many half of the project.

The rules engine harmonizes a melody in Bach's language, where there is a
strong notion of correct and search wins. Reharmonization has no correct
answer — five substitutions of the same bar can all be valid and differ
entirely in character — so the argmax of a reharmonization distribution is by
construction the boring one. That is the structural reason this package
samples where the rules engine searches.

Importing `ml.reharm.engine` registers the `jazz_reharm` engine.
"""
