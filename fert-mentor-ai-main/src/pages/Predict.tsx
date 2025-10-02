import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Sprout, Droplets, Thermometer, Cloud } from "lucide-react";
import { toast } from "sonner";
import { fetchMetadata, predictFertilizer, type PredictRequest, type PredictResponse } from "@/lib/api";

const numberFields = [
  "nitrogen",
  "phosphorus",
  "potassium",
  "temperature",
  "humidity",
  "rainfall",
  "moisture",
  "ph",
] as const;

const Predict = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    nitrogen: "",
    phosphorus: "",
    potassium: "",
    temperature: "",
    humidity: "",
    ph: "",
    rainfall: "",
    moisture: "",
    crop: ""
  });
  const [cropOptions, setCropOptions] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  useEffect(() => {
    let abort = new AbortController();
    fetchMetadata(abort.signal)
      .then(meta => {
        setCropOptions(meta.crops || []);
      })
      .catch(err => {
        setErrorMsg(err.message || "Failed to load crops");
      });
    return () => abort.abort();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMsg(null);
    setResult(null);

    // Validate all fields are filled
    const isEmpty = Object.values(formData).some(val => val === "");
    if (isEmpty) {
      toast.error("Please fill in all fields");
      return;
    }

    // Convert and map to backend payload
    const payload: PredictRequest = {
      N: Number(formData.nitrogen),
      P: Number(formData.phosphorus),
      K: Number(formData.potassium),
      temperature: Number(formData.temperature),
      humidity: Number(formData.humidity),
      pH: Number(formData.ph),
      rainfall: Number(formData.rainfall),
      moisture: Number(formData.moisture),
      crop: formData.crop,
    };

    // Basic numeric validation
    const hasNaN = (Object.entries(payload) as [string, any][]).some(([k, v]) => {
      if (k === "crop") return false;
      return typeof v !== "number" || Number.isNaN(v);
    });
    if (hasNaN) {
      toast.error("Please enter valid numbers in all numeric fields");
      return;
    }

    try {
      setIsLoading(true);
      const res = await predictFertilizer(payload);
      setResult(res);
      toast.success("Recommendation generated");
    } catch (err: any) {
      setErrorMsg(err?.message || "Prediction failed");
      toast.error("Prediction failed");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/30">
      {/* Header */}
      <header className="border-b bg-card/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sprout className="h-8 w-8 text-primary" />
            <h1 className="text-2xl font-bold text-foreground">FertiSmart</h1>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" onClick={() => navigate("/reports")}>
              Reports
            </Button>
            <Button variant="outline" onClick={() => navigate("/fertilizers")}>
              Browse Fertilizers
            </Button>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="w-full px-4 py-12">
        <div className="text-center max-w-3xl mx-auto mb-12">
          <h2 className="text-4xl md:text-5xl lg:text-6xl xl:text-7xl font-bold mb-4 bg-gradient-to-r from-primary via-primary to-secondary bg-clip-text text-transparent">
            ML-Powered Fertilizer Recommendation
          </h2>
          <p className="text-lg md:text-xl lg:text-2xl text-muted-foreground">
            Get precise fertilizer recommendations based on your soil characteristics and crop requirements
          </p>
        </div>

        {/* Input Form */}
        <Card className="w-full max-w-6xl mx-auto shadow-[var(--shadow-card)] border-border/50">
          <CardHeader>
            <CardTitle className="text-2xl md:text-3xl lg:text-4xl flex items-center gap-2">
              <Sprout className="h-6 w-6 text-primary" />
              Enter Soil & Crop Details
            </CardTitle>
            <CardDescription className="text-base md:text-lg">
              Provide accurate information for the best fertilizer recommendation
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* NPK Section */}
              <div className="bg-muted/30 p-6 rounded-lg space-y-4">
                <h3 className="font-semibold text-lg md:text-xl lg:text-2xl flex items-center gap-2 text-foreground">
                  <div className="h-2 w-2 rounded-full bg-primary"></div>
                  Soil Nutrients (NPK)
                </h3>
                <div className="grid md:grid-cols-3 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="nitrogen">Nitrogen (N) - kg/ha</Label>
                    <Input
                      id="nitrogen"
                      type="number"
                      placeholder="e.g., 40"
                      value={formData.nitrogen}
                      onChange={(e) => handleInputChange("nitrogen", e.target.value)}
                      className="bg-background"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="phosphorus">Phosphorus (P) - kg/ha</Label>
                    <Input
                      id="phosphorus"
                      type="number"
                      placeholder="e.g., 30"
                      value={formData.phosphorus}
                      onChange={(e) => handleInputChange("phosphorus", e.target.value)}
                      className="bg-background"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="potassium">Potassium (K) - kg/ha</Label>
                    <Input
                      id="potassium"
                      type="number"
                      placeholder="e.g., 20"
                      value={formData.potassium}
                      onChange={(e) => handleInputChange("potassium", e.target.value)}
                      className="bg-background"
                    />
                  </div>
                </div>
              </div>

              {/* Environmental Conditions */}
              <div className="bg-muted/30 p-6 rounded-lg space-y-4">
                <h3 className="font-semibold text-lg md:text-xl lg:text-2xl flex items-center gap-2 text-foreground">
                  <div className="h-2 w-2 rounded-full bg-secondary"></div>
                  Environmental Conditions
                </h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="temperature" className="flex items-center gap-2">
                      <Thermometer className="h-4 w-4" />
                      Temperature (°C)
                    </Label>
                    <Input
                      id="temperature"
                      type="number"
                      step="0.1"
                      placeholder="e.g., 25.5"
                      value={formData.temperature}
                      onChange={(e) => handleInputChange("temperature", e.target.value)}
                      className="bg-background"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="humidity" className="flex items-center gap-2">
                      <Droplets className="h-4 w-4" />
                      Humidity (%)
                    </Label>
                    <Input
                      id="humidity"
                      type="number"
                      step="0.1"
                      placeholder="e.g., 65.5"
                      value={formData.humidity}
                      onChange={(e) => handleInputChange("humidity", e.target.value)}
                      className="bg-background"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="rainfall" className="flex items-center gap-2">
                      <Cloud className="h-4 w-4" />
                      Rainfall (mm)
                    </Label>
                    <Input
                      id="rainfall"
                      type="number"
                      step="0.1"
                      placeholder="e.g., 150.5"
                      value={formData.rainfall}
                      onChange={(e) => handleInputChange("rainfall", e.target.value)}
                      className="bg-background"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="moisture">Soil Moisture (%)</Label>
                    <Input
                      id="moisture"
                      type="number"
                      step="0.1"
                      placeholder="e.g., 45.5"
                      value={formData.moisture}
                      onChange={(e) => handleInputChange("moisture", e.target.value)}
                      className="bg-background"
                    />
                  </div>
                </div>
              </div>

              {/* Soil pH & Crop */}
              <div className="grid md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="ph">Soil pH Level</Label>
                  <Input
                    id="ph"
                    type="number"
                    step="0.1"
                    placeholder="e.g., 6.5"
                    value={formData.ph}
                    onChange={(e) => handleInputChange("ph", e.target.value)}
                    className="bg-background"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="crop">Crop Type</Label>
                  <Select value={formData.crop} onValueChange={(value) => handleInputChange("crop", value)}>
                    <SelectTrigger className="bg-background">
                      <SelectValue placeholder="Select crop" />
                    </SelectTrigger>
                    <SelectContent>
                      {cropOptions.map(crop => (
                        <SelectItem key={crop} value={crop}>
                          {crop}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <Button 
                type="submit" 
                className="w-full h-12 text-lg font-semibold bg-gradient-to-r from-primary to-primary/80 hover:from-primary/90 hover:to-primary/70 shadow-[var(--shadow-soft)]"
                disabled={isLoading}
              >
                {isLoading ? "Predicting..." : "Get Fertilizer Recommendation"}
              </Button>
              {errorMsg && (
                <p className="text-sm text-red-500 mt-2">{errorMsg}</p>
              )}
              {result && (
                <div className="mt-4 p-4 border rounded-lg bg-muted/20">
                  <p className="text-sm text-muted-foreground">Recommended fertilizer</p>
                  <p className="text-xl font-semibold">{result.fertilizer}</p>
                  <p className="text-xs text-muted-foreground">Class: {result.predicted_class}</p>
                </div>
              )}
            </form>
          </CardContent>
        </Card>

        {/* Info Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mt-12 w-full">
          <Card className="border-primary/20 hover:border-primary/40 transition-[var(--transition-smooth)]">
            <CardHeader>
              <Sprout className="h-8 w-8 text-primary mb-2" />
              <CardTitle className="text-lg md:text-xl lg:text-2xl">Precise Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm md:text-base lg:text-lg text-muted-foreground">
                AI-powered recommendations based on comprehensive soil and environmental data
              </p>
            </CardContent>
          </Card>
          <Card className="border-secondary/20 hover:border-secondary/40 transition-[var(--transition-smooth)]">
            <CardHeader>
              <Droplets className="h-8 w-8 text-secondary mb-2" />
              <CardTitle className="text-lg">Optimize Yield</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Maximize crop productivity while minimizing fertilizer costs and environmental impact
              </p>
            </CardContent>
          </Card>
          <Card className="border-accent/20 hover:border-accent/40 transition-[var(--transition-smooth)]">
            <CardHeader>
              <Thermometer className="h-8 w-8 text-accent mb-2" />
              <CardTitle className="text-lg">Sustainable Farming</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Support eco-friendly practices with optimal fertilizer application recommendations
              </p>
            </CardContent>
          </Card>
        </div>
      </section>
    </div>
  );
};

export default Predict;
