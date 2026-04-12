'use client';

import { cn } from "../../lib/utils";
import { useState } from 'react';

interface FlipTextProps {
    word?: string;
    className?: string;
}

export default function FlipTextReveal({ word = "DIGITAL REALITY", className = "" }: FlipTextProps) {
    const [key, setKey] = useState(0);

    const replay = () => {
        setKey((prev) => prev + 1);
    };

    return (
        <div className={`flip-container ${className}`}>
            <div key={key} className="text-wrapper">
                <h1 className="title" aria-label={word}>
                    {word.split("").map((char, i) => (
                        <span
                            key={`${key}-${i}`}
                            className="flip-char"
                            style={{ "--index": i } as React.CSSProperties}
                        >
                            {char === " " ? "\u00A0" : char}
                        </span>
                    ))}
                </h1>
            </div>

            <button className="replay-button" onClick={replay}>
                <span className="btn-text">Replay Action</span>
            </button>

        </div>
    );
}
