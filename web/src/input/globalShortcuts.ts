import { MUSICAL_TYPING_KEY_SET } from "./musicalTyping";

export const CHARACTER_SHORTCUTS = [
  { key: "/", action: "loop" },
  { key: "m", action: "metronome" },
  { key: "r", action: "record" },
  { key: "1", action: "result-a" },
  { key: "2", action: "result-b" },
] as const;

const collisions = CHARACTER_SHORTCUTS.filter(({ key }) =>
  MUSICAL_TYPING_KEY_SET.has(key),
);
if (collisions.length > 0) {
  throw new Error(
    `Global shortcuts collide with musical typing: ${collisions
      .map(({ key }) => key)
      .join(", ")}`,
  );
}
