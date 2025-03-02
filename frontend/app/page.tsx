"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import DrawingCanvas from "@/components/DrawingCanvas";
import LabelSelector from "@/components/LabelSelector";

export default function Home() {
  const [currentLabel, setCurrentLabel] = useState<number>(1);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);

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
      <h1 className="text-3xl font-bold mb-8">Kannada Alphabet Drawing</h1>
      
      <div className="flex flex-col md:flex-row w-full max-w-6xl gap-6">
        {/* Left side - Label display */}
        <Card className="flex-1 p-6 flex flex-col items-center justify-center">
          <h2 className="text-xl font-semibold mb-4">Current Label</h2>
          <div className="text-9xl font-bold mb-8">{currentLabel}</div>
          <LabelSelector 
            currentLabel={currentLabel} 
            setCurrentLabel={setCurrentLabel} 
          />
        </Card>

        {/* Right side - Drawing canvas */}
        <Card className="flex-1 p-6">
          <h2 className="text-xl font-semibold mb-4">Draw Here</h2>
          <DrawingCanvas 
            onSubmit={handleSubmit}
            isSubmitting={isSubmitting}
          />
        </Card>
      </div>
    </main>
  );
}