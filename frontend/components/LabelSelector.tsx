"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { 
  ChevronLeft, 
  ChevronRight, 
  ChevronsLeft, 
  ChevronsRight 
} from "lucide-react";

interface LabelSelectorProps {
  currentLabel: number;
  setCurrentLabel: (label: number) => void;
}

export default function LabelSelector({ 
  currentLabel, 
  setCurrentLabel 
}: LabelSelectorProps) {
  const [inputValue, setInputValue] = useState<string>(currentLabel.toString());
  
  const MIN_LABEL = 1;
  const MAX_LABEL = 50; // Assuming 50 Kannada alphabets/characters
  
  const handlePrevious = () => {
    if (currentLabel > MIN_LABEL) {
      const newLabel = currentLabel - 1;
      setCurrentLabel(newLabel);
      setInputValue(newLabel.toString());
    }
  };
  
  const handleNext = () => {
    if (currentLabel < MAX_LABEL) {
      const newLabel = currentLabel + 1;
      setCurrentLabel(newLabel);
      setInputValue(newLabel.toString());
    }
  };
  
  const handleFirst = () => {
    setCurrentLabel(MIN_LABEL);
    setInputValue(MIN_LABEL.toString());
  };
  
  const handleLast = () => {
    setCurrentLabel(MAX_LABEL);
    setInputValue(MAX_LABEL.toString());
  };
  
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  };
  
  const handleInputBlur = () => {
    const numValue = parseInt(inputValue);
    if (!isNaN(numValue) && numValue >= MIN_LABEL && numValue <= MAX_LABEL) {
      setCurrentLabel(numValue);
    } else {
      setInputValue(currentLabel.toString());
    }
  };
  
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      handleInputBlur();
    }
  };
  
  return (
    <div className="flex flex-col items-center w-full">
      <div className="flex items-center justify-between w-full max-w-xs mb-2">
        <Button 
          variant="outline" 
          size="icon" 
          onClick={handleFirst}
          disabled={currentLabel === MIN_LABEL}
        >
          <ChevronsLeft className="h-4 w-4" />
        </Button>
        
        <Button 
          variant="outline" 
          size="icon" 
          onClick={handlePrevious}
          disabled={currentLabel === MIN_LABEL}
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>
        
        <div className="w-20">
          <Input
            type="text"
            value={inputValue}
            onChange={handleInputChange}
            onBlur={handleInputBlur}
            onKeyDown={handleKeyDown}
            className="text-center"
          />
        </div>
        
        <Button 
          variant="outline" 
          size="icon" 
          onClick={handleNext}
          disabled={currentLabel === MAX_LABEL}
        >
          <ChevronRight className="h-4 w-4" />
        </Button>
        
        <Button 
          variant="outline" 
          size="icon" 
          onClick={handleLast}
          disabled={currentLabel === MAX_LABEL}
        >
          <ChevronsRight className="h-4 w-4" />
        </Button>
      </div>
      
      <div className="text-sm text-muted-foreground">
        Label {currentLabel} of {MAX_LABEL}
      </div>
    </div>
  );
}