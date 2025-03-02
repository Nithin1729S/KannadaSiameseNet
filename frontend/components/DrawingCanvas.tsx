"use client";

import { useRef, useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Pencil } from "lucide-react";

interface DrawingCanvasProps {
  onSubmit: (imageData: string) => void;
  isSubmitting: boolean;
}

export default function DrawingCanvas({ onSubmit, isSubmitting }: DrawingCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [tool, setTool] = useState<"pencil" | "eraser">("pencil");
  
  // Initialize canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    
    // Set canvas background to white
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Set initial drawing style
    ctx.lineWidth = 5;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "black";
  }, []);
  
  // Drawing functions
  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    
    setIsDrawing(true);
    
    // Set drawing style based on selected tool
    if (tool === "pencil") {
      ctx.strokeStyle = "black";
    } else {
      ctx.strokeStyle = "white";
    }
    
    // Get coordinates
    let x, y;
    if ('touches' in e) {
      const rect = canvas.getBoundingClientRect();
      x = e.touches[0].clientX - rect.left;
      y = e.touches[0].clientY - rect.top;
    } else {
      x = e.nativeEvent.offsetX;
      y = e.nativeEvent.offsetY;
    }
    
    ctx.beginPath();
    ctx.moveTo(x, y);
  };
  
  const draw = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    
    // Get coordinates
    let x, y;
    if ('touches' in e) {
      const rect = canvas.getBoundingClientRect();
      x = e.touches[0].clientX - rect.left;
      y = e.touches[0].clientY - rect.top;
    } else {
      x = e.nativeEvent.offsetX;
      y = e.nativeEvent.offsetY;
    }
    
    ctx.lineTo(x, y);
    ctx.stroke();
  };
  
  const stopDrawing = () => {
    setIsDrawing(false);
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    
    ctx.closePath();
  };
  
  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  };
  
  const handleSubmit = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    // Create a temporary canvas for resizing
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = 105;
    tempCanvas.height = 105;
    const tempCtx = tempCanvas.getContext("2d");
    
    if (!tempCtx) return;
    
    // Draw the original canvas content onto the temporary canvas (resizing it)
    tempCtx.fillStyle = "white";
    tempCtx.fillRect(0, 0, 105, 105);
    tempCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, 105, 105);
    
    // Get the image data as base64 string
    const imageData = tempCanvas.toDataURL("image/png");
    
    // Send to parent component
    onSubmit(imageData);
  };
  
  return (
    <div className="flex flex-col items-center">
      <div className="mb-4 w-full flex justify-between items-center">
        <ToggleGroup type="single" value={tool} onValueChange={(value) => value && setTool(value as "pencil" | "eraser")}>
          <ToggleGroupItem value="pencil" aria-label="Write">
            <Pencil className="h-4 w-4" />
          </ToggleGroupItem>
        </ToggleGroup>
        
        <Button variant="outline" onClick={clearCanvas}>
          Clear
        </Button>
      </div>
      
      <div className="border border-gray-300 rounded-md overflow-hidden touch-none">
        <canvas
          ref={canvasRef}
          width={300}
          height={300}
          className="bg-white touch-none"
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing}
          onTouchStart={startDrawing}
          onTouchMove={draw}
          onTouchEnd={stopDrawing}
        />
      </div>
      
      <Button 
        className="mt-4 w-full" 
        onClick={handleSubmit}
        disabled={isSubmitting}
      >
        {isSubmitting ? "Submitting..." : "Submit"}
      </Button>
    </div>
  );
}