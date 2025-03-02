"use client";

import { useState,useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import DrawingCanvas from "@/components/DrawingCanvas";
import LabelSelector from "@/components/LabelSelector";

const kannadaAlphabets:string[] = [
  'ಅ', 'ಆ', 'ಇ', 'ಈ', 'ಉ', 'ಊ', 'ಋ', 'ೠ','ಎ', 'ಏ', 'ಐ',
  'ಒ', 'ಓ', 'ಔ','ಅಂ','ಅಃ', 'ಕ', 'ಖ', 'ಗ', 'ಘ', 'ಙ', 'ಚ', 'ಛ',
  'ಜ', 'ಝ', 'ಞ', 'ಟ', 'ಠ', 'ಡ', 'ಢ', 'ಣ', 'ತ', 'ಥ',
  'ದ', 'ಧ', 'ನ', 'ಪ', 'ಫ', 'ಬ', 'ಭ', 'ಮ', 'ಯ', 'ರ',
  'ಲ', 'ವ', 'ಶ', 'ಷ', 'ಸ', 'ಹ', 'ಳ'
];


export default function Home() {
  const [currentLabel, setCurrentLabel] = useState<number>(0);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const clearCanvasRef = useRef<(() => void) | null>(null);
  const canvasRef = useRef<{ clearCanvas: () => void } | null>(null);
  const handleLabelChange = (newLabel: number) => {
    setCurrentLabel(newLabel);
    // Clear the canvas when label changes
    if (canvasRef.current) {
      canvasRef.current.clearCanvas();
    }
  };

  const handleSubmit = async (imageData: string) => {
    setIsSubmitting(true);
    try {
      // Replace with your actual API endpoint
      const response = await fetch("/api/submit", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          label: currentLabel,
          image: imageData,
        }),
      });

      if (response.ok) {
        console.log("Successfully submitted drawing");
        // You can add success notification here
      } else {
        console.error("Failed to submit drawing");
        // You can add error notification here
      }
    } catch (error) {
      console.error("Error submitting drawing:", error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center p-4 md:p-8">
      <h1 className="text-3xl font-bold mb-8">Kannada Alphabet Recognition</h1>
      
      <div className="flex flex-col md:flex-row w-full max-w-6xl gap-6">
        {/* Left side - Label display */}
        <Card className="flex-1 p-6 flex flex-col items-center justify-center">
          <h2 className="text-xl font-semibold mb-4">Current Label</h2>
          <div className="text-9xl font-bold mb-8">{kannadaAlphabets[currentLabel]}</div>
          <LabelSelector 
            currentLabel={currentLabel} 
            setCurrentLabel={handleLabelChange} 
          />
        </Card>

        {/* Right side - Drawing canvas */}
        <Card className="flex-1 p-6">
          <h2 className="text-xl font-semibold mb-4">Draw Here</h2>
          <DrawingCanvas 
            onSubmit={handleSubmit}
            isSubmitting={isSubmitting}
            ref={canvasRef}
          />
        </Card>
      </div>
    </main>
  );
}