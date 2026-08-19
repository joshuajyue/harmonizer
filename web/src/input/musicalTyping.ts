export const MUSICAL_TYPING_KEYS = [
  "a",
  "w",
  "s",
  "e",
  "d",
  "f",
  "t",
  "g",
  "y",
  "h",
  "u",
  "j",
  "k",
  "o",
  "l",
  "p",
  ";",
] as const;

export const MUSICAL_TYPING_OFFSETS = new Map<string, number>(
  MUSICAL_TYPING_KEYS.map((key, index) => [key, index]),
);

export const MUSICAL_TYPING_KEY_SET = new Set<string>(
  MUSICAL_TYPING_KEYS,
);
