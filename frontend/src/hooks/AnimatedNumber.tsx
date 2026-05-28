// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

import { useMemo } from "react";
import { useAnimatedNumber } from "./useAnimatedNumber";
import "../styles/hooks/animatedNumber.css";

const DIGIT_EM = 0.62;
const SEP_EM = 0.35;
const DIGITS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] as const;

interface Props {
  value: number;
  format?: (v: number) => string;
  className?: string;
  /** Max pixel width to fit within. Font size scales down to fit. */
  fitWidth?: number;
  /** Largest font size allowed (px). Default 34. */
  maxFontSize?: number;
  /** Smallest font size allowed (px). Default 16. */
  minFontSize?: number;
}

function computeFontSize(
  text: string,
  fitWidth: number,
  maxFs: number,
  minFs: number,
): number {
  let digitCount = 0;
  let sepCount = 0;
  for (const ch of text) {
    if (ch >= "0" && ch <= "9") digitCount++;
    else sepCount++;
  }
  const totalEm = digitCount * DIGIT_EM + sepCount * SEP_EM;
  if (totalEm <= 0) return maxFs;
  const ideal = fitWidth / totalEm;
  return Math.max(minFs, Math.min(maxFs, Math.floor(ideal)));
}

export function AnimatedNumber({
  value,
  format,
  className,
  fitWidth,
  maxFontSize = 34,
  minFontSize = 16,
}: Props) {
  const animated = useAnimatedNumber(value);
  const text = useMemo(() => {
    if (format) return format(animated);
    return Number.isInteger(value)
      ? Math.round(animated).toString()
      : animated.toFixed(1);
  }, [animated, format, value]);

  const sizeStyle = useMemo(() => {
    if (!fitWidth) return undefined;
    const fs = computeFontSize(
      text, fitWidth, maxFontSize, minFontSize,
    );
    return { fontSize: fs } as const;
  }, [text.length, fitWidth, maxFontSize, minFontSize]);

  return (
    <span
      className={`odometer ${className ?? ""}`}
      aria-label={String(value)}
      style={sizeStyle}
    >
      {text.split("").map((ch, i) => {
        const isDigit = ch >= "0" && ch <= "9";
        if (!isDigit) {
          return (
            <span key={`s-${i}`} className="odometer__sep">
              {ch}
            </span>
          );
        }
        return (
          <span key={`d-${i}`} className="odometer__slot">
            <span
              className="odometer__reel"
              style={{
                transform: `translateY(${-Number(ch) * 10}%)`,
              }}
            >
              {DIGITS.map((d) => (
                <span key={d} className="odometer__digit">
                  {d}
                </span>
              ))}
            </span>
          </span>
        );
      })}
    </span>
  );
}
