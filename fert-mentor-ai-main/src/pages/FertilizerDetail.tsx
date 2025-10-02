import { useParams, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Sprout, ArrowLeft, Leaf, Droplets, Zap, Shield, Sun, AlertCircle, CheckCircle2, TrendingUp, Beaker } from "lucide-react";

const fertilizerData: Record<string, any> = {
  "10-26-26": {
    name: "10-26-26",
    formula: "NPK 10-26-26",
    icon: Zap,
    color: "primary",
    composition: { nitrogen: "10%", phosphorus: "26% P₂O₅", potassium: "26% K₂O" },
    description: "Complex NPK with high P and K for basal application, strong roots, and improved flowering/fruiting. Balanced nutrition emphasizing reproductive stages.",
    characteristics: [
      "Granular 1-4mm",
      "High water solubility",
      "Non-hygroscopic",
      "62% total nutrients"
    ],
    benefits: [
      "Promotes strong root development",
      "Enhances flower and fruit development",
      "Improves crop quality and yield",
      "Supports energy transfer and metabolism",
      "Increases disease resistance and stress tolerance"
    ],
    applicationMethod: [
      "Broadcast and incorporate",
      "Basal at sowing",
      "Split application",
      "Suitable for fertigation"
    ],
    dosage: "100-150 kg/ha for field crops; 50-100 kg/ha for vegetables",
    suitableFor: ["Cotton", "Groundnut", "Soybean", "Chilli", "Potato", "Vegetables", "Wheat"],
    bestTime: "At planting as basal; combine with urea for top-dress N",
    precautions: [
      "Store dry, cool, away from sunlight",
      "Keep from combustibles and heat",
      "Calibrate equipment for even application",
      "Monitor soil pH; P availability drops in alkaline soils"
    ],
    soilRequirements: { ph: "6.0-7.5", moisture: "Well-drained", temperature: "20-27°C" }
  },
  "14-35-14": {
    name: "14-35-14",
    formula: "NPK 14-35-14",
    icon: Droplets,
    color: "secondary",
    composition: { nitrogen: "14%", phosphorus: "35% P₂O₅", potassium: "14% K₂O" },
    description: "High-phosphorus complex (63% total nutrients) with added K; excellent for P-demanding crops.",
    characteristics: [
      "63% total nutrients",
      "Ammoniacal N for sustained release",
      "Water-soluble P and K",
      "Neutral nature"
    ],
    benefits: [
      "Highest P content among complexes",
      "Vigorous root development",
      "Enhances flowering and fruiting",
      "Improves stress tolerance and disease resistance",
      "Supports DNA and RNA formation"
    ],
    applicationMethod: [
      "Basal at planting",
      "Broadcast with incorporation",
      "Band placement near rows",
      "Foliar compatible at 3-5 g/L"
    ],
    dosage: "100-150 kg/ha; foliar 3-5 g/L",
    suitableFor: ["Cotton", "Groundnut", "Chilli", "Soybean", "Potato", "Pulses", "Vegetables"],
    bestTime: "At sowing (basal); early vegetative for foliar",
    precautions: [
      "Not for chloride-sensitive crops (tobacco, grapes)",
      "Store away from urea/combustibles",
      "Maintain spacing from seeds",
      "Ensure ventilation in storage"
    ],
    soilRequirements: { ph: "6.0-7.5", moisture: "Various conditions", temperature: "25-26°C" }
  },
  "17-17-17": {
    name: "17-17-17",
    formula: "NPK 17-17-17",
    icon: Shield,
    color: "accent",
    composition: { nitrogen: "17%", phosphorus: "17% P₂O₅", potassium: "17% K₂O" },
    description: "Balanced NPK supporting all phases from vegetative growth to fruiting; universal application.",
    characteristics: [
      "Equal NPK 1:1:1",
      "Granular 2-4mm",
      "High water solubility",
      "Often enhanced with micronutrients"
    ],
    benefits: [
      "Balanced nutrition for all stages",
      "Strong roots and leaf growth",
      "Enhances flowering, fruiting, quality",
      "Improves drought tolerance and disease resistance"
    ],
    applicationMethod: [
      "Broadcast 100-150 kg/ha",
      "Row application 50-100 kg/ha",
      "Fertigation",
      "Foliar 250-300 g per 100 L"
    ],
    dosage: "50-75 kg/acre; 25-50 kg/ha for balanced fertigation",
    suitableFor: ["Pomegranate", "Orange", "Oil seeds", "Vegetables", "All crops"],
    bestTime: "Planting, vegetative and fruiting stages",
    precautions: [
      "Avoid over-application to prevent imbalance",
      "Store dry, ventilated",
      "Use calibrated equipment",
      "Monitor soil conditions regularly"
    ],
    soilRequirements: { ph: "6.5-7.0", moisture: "Well-drained with retention", temperature: "24-25°C" }
  },
  "28-28": {
    name: "28-28",
    formula: "NP 28-28",
    icon: Zap,
    color: "primary",
    composition: { nitrogen: "28%", phosphorus: "28% P₂O₅", potassium: "0% K₂O" },
    description: "High-analysis water-soluble NP providing rapid vegetative growth; ideal during peak growth, no K.",
    characteristics: [
      "100% water-soluble powder",
      "N:P ratio 1:1",
      "Compatible with common pesticides",
      "Controlled release properties"
    ],
    benefits: [
      "Quick vegetative growth and tillering",
      "Vigorous branching and foliage",
      "Instant nutrition in peak growth",
      "Excellent for foliar and fertigation",
      "Boosts growth in dry or excess rainfall"
    ],
    applicationMethod: [
      "Foliar 1% (10 g/L)",
      "Fertigation 2-3 kg/acre",
      "Direct soil application in vegetative stage",
      "Drip compatible"
    ],
    dosage: "Foliar 10 g/L; fertigation 2-3 kg/acre",
    suitableFor: ["Pomegranate", "Mango", "Orange", "Oil seeds", "Cereals", "Vegetables"],
    bestTime: "20-60 days after planting",
    precautions: [
      "Store clean and dry",
      "Avoid mixing with calcium-based fertilizers",
      "Use protective equipment",
      "Ensure K from other sources"
    ],
    soilRequirements: { ph: "6.0-7.0", moisture: "Dry and wet conditions", temperature: "25-26°C" }
  },
  dap: {
    name: "DAP (Diammonium Phosphate)",
    formula: "(NH₄)₂HPO₄",
    icon: Droplets,
    color: "secondary",
    composition: { nitrogen: "18%", phosphorus: "46% P₂O₅", potassium: "0%", sulfur: "1.5%" },
    description: "Widely used P fertilizer with N and P; excellent starter for root development and legumes.",
    characteristics: [
      "High solubility (588 g/L at 20°C)",
      "Granular with good physical properties",
      "pH neutral to slightly alkaline (7.5-8.0)",
      "Non-hygroscopic"
    ],
    benefits: [
      "Excellent source of readily available P",
      "Enhances root development and early growth",
      "Improves nitrogen fixation in legumes",
      "Supports energy transfer and metabolic processes",
      "Quick dissolution and availability"
    ],
    applicationMethod: [
      "Basal before sowing",
      "Band placement near seed rows",
      "Suitable for fertigation",
      "Pre-sowing soil incorporation"
    ],
    dosage: "Varies by crop and soil P status; full P dose at planting",
    suitableFor: ["Pulses", "Mango", "Grapes", "Apple", "Starter fertilizer"],
    bestTime: "At sowing as basal; combine with urea for N needs",
    precautions: [
      "Avoid mixing with alkaline materials (ammonia release)",
      "Store away from heat and combustibles",
      "Monitor thermal decomposition above 70°C",
      "Ensure adequate potassium from other sources"
    ],
    soilRequirements: { ph: "6.0-7.5", moisture: "Well-drained; avoid waterlogging", temperature: "Normal growing temperatures" }
  },
  urea: {
    name: "Urea",
    formula: "CO(NH₂)₂",
    icon: Leaf,
    color: "primary",
    composition: { nitrogen: "46%", phosphorus: "0%", potassium: "0%" },
    description: "Most widely used nitrogen fertilizer; 46% N for strong vegetative growth; suitable for most crops.",
    characteristics: [
      "White crystalline solid",
      "High water solubility",
      "Converts to ammonium within 48 hours",
      "Low cost per unit N"
    ],
    benefits: [
      "Highest N concentration among solids",
      "Promotes vigorous vegetative growth",
      "Cost-effective and versatile",
      "Essential for protein and chlorophyll formation"
    ],
    applicationMethod: [
      "Broadcast with immediate incorporation",
      "Soil injection or banding",
      "Fertigation",
      "Foliar spray when dissolved"
    ],
    dosage: "Varies by crop N requirement and soil status; split applications recommended",
    suitableFor: ["Paddy", "Maize", "Cotton", "Wheat", "Sugarcane", "Vegetables"],
    bestTime: "Pre-plant incorporation or side-dress during active growth",
    precautions: [
      "Incorporate to prevent volatilization",
      "Avoid surface application without water",
      "Store dry away from moisture",
      "Can volatilize ammonia if left on surface"
    ],
    soilRequirements: { ph: "6.0-7.5", moisture: "Needs moisture/irrigation for activation", temperature: "All ranges" }
  }
};

const FertilizerDetail = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  
  const fertilizer = id ? fertilizerData[id] : null;

  if (!fertilizer) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <Card className="max-w-md">
          <CardHeader>
            <CardTitle>Fertilizer Not Found</CardTitle>
            <CardDescription>The requested fertilizer information is not available.</CardDescription>
          </CardHeader>
          <CardContent>
            <Button onClick={() => navigate("/fertilizers")}>
              Back to Fertilizers
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  const Icon = fertilizer.icon;

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/30">
      {/* Header */}
      <header className="border-b bg-card/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2 cursor-pointer" onClick={() => navigate("/")}>
            <Sprout className="h-8 w-8 text-primary" />
            <h1 className="text-2xl font-bold text-foreground">FertiSmart</h1>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" onClick={() => navigate("/reports")}>
              Reports
            </Button>
            <Button variant="outline" onClick={() => navigate("/fertilizers")}>
              <ArrowLeft className="mr-2 h-4 w-4" />
              All Fertilizers
            </Button>
          </div>
        </div>
      </header>

      <div className="w-full px-4 py-12">
        {/* Header Section */}
        <div className="mb-8">
          <div className="flex items-start gap-6 mb-6">
            <div className={`p-4 rounded-xl bg-${fertilizer.color}/10`}>
              <Icon className={`h-12 w-12 text-${fertilizer.color}`} />
            </div>
            <div className="flex-1">
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold mb-2 text-foreground">{fertilizer.name}</h1>
              <p className="text-xl md:text-2xl lg:text-3xl text-muted-foreground font-mono mb-4">{fertilizer.formula}</p>
              <div className="flex flex-wrap gap-2">
                {fertilizer.suitableFor.map((crop: string) => (
                  <Badge key={crop} variant="secondary">
                    {crop}
                  </Badge>
                ))}
              </div>
            </div>
          </div>
          <p className="text-lg md:text-xl lg:text-2xl text-muted-foreground leading-relaxed">{fertilizer.description}</p>
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-6">
            {/* Composition */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg md:text-xl lg:text-2xl flex items-center gap-2">
                  <Beaker className="h-5 w-5 text-primary" />
                  Nutrient Composition
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-3 gap-4">
                  <div className="text-center p-4 bg-muted/30 rounded-lg">
                    <p className="text-sm text-muted-foreground mb-1">Nitrogen (N)</p>
                    <p className="text-2xl font-bold text-primary">{fertilizer.composition.nitrogen}</p>
                  </div>
                  <div className="text-center p-4 bg-muted/30 rounded-lg">
                    <p className="text-sm text-muted-foreground mb-1">Phosphorus (P)</p>
                    <p className="text-2xl font-bold text-secondary">{fertilizer.composition.phosphorus}</p>
                  </div>
                  <div className="text-center p-4 bg-muted/30 rounded-lg">
                    <p className="text-sm text-muted-foreground mb-1">Potassium (K)</p>
                    <p className="text-2xl font-bold text-accent">{fertilizer.composition.potassium}</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Benefits */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle2 className="h-5 w-5 text-primary" />
                  Key Benefits
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-3">
                  {fertilizer.benefits.map((benefit: string, idx: number) => (
                    <li key={idx} className="flex items-start gap-3">
                      <div className="h-6 w-6 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0 mt-0.5">
                        <CheckCircle2 className="h-4 w-4 text-primary" />
                      </div>
                      <span className="text-muted-foreground">{benefit}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>

            {/* Characteristics */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="h-5 w-5 text-primary" />
                  Characteristics
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-3">
                  {fertilizer.characteristics.map((char: string, idx: number) => (
                    <div key={idx} className="flex items-center gap-2 p-3 bg-muted/30 rounded-lg">
                      <div className="h-2 w-2 rounded-full bg-primary"></div>
                      <span className="text-sm text-foreground">{char}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Application Method */}
            <Card>
              <CardHeader>
                <CardTitle>Application Methods</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2">
                  {fertilizer.applicationMethod.map((method: string, idx: number) => (
                    <li key={idx} className="flex items-center gap-3 text-muted-foreground">
                      <div className="h-1.5 w-1.5 rounded-full bg-primary"></div>
                      {method}
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Dosage & Timing */}
            <Card className="border-primary/20">
              <CardHeader>
                <CardTitle className="text-lg">Recommended Dosage</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <p className="text-sm font-semibold text-foreground mb-1">Amount</p>
                  <p className="text-sm text-muted-foreground">{fertilizer.dosage}</p>
                </div>
                <Separator />
                <div>
                  <p className="text-sm font-semibold text-foreground mb-1">Best Time</p>
                  <p className="text-sm text-muted-foreground">{fertilizer.bestTime}</p>
                </div>
              </CardContent>
            </Card>

            {/* Soil Requirements */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Soil Requirements</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div>
                  <p className="text-xs font-semibold text-muted-foreground mb-1">pH Level</p>
                  <p className="text-sm text-foreground">{fertilizer.soilRequirements.ph}</p>
                </div>
                <Separator />
                <div>
                  <p className="text-xs font-semibold text-muted-foreground mb-1">Moisture</p>
                  <p className="text-sm text-foreground">{fertilizer.soilRequirements.moisture}</p>
                </div>
                <Separator />
                <div>
                  <p className="text-xs font-semibold text-muted-foreground mb-1">Temperature</p>
                  <p className="text-sm text-foreground">{fertilizer.soilRequirements.temperature}</p>
                </div>
              </CardContent>
            </Card>

            {/* Precautions */}
            <Card className="border-destructive/20">
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <AlertCircle className="h-5 w-5 text-destructive" />
                  Precautions
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2">
                  {fertilizer.precautions.map((precaution: string, idx: number) => (
                    <li key={idx} className="flex items-start gap-2 text-sm text-muted-foreground">
                      <AlertCircle className="h-4 w-4 text-destructive flex-shrink-0 mt-0.5" />
                      <span>{precaution}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>

            {/* CTA */}
            <Card className="bg-gradient-to-br from-primary/5 to-secondary/5 border-primary/20">
              <CardContent className="pt-6">
                <p className="text-sm text-muted-foreground mb-4">
                  Need help determining if this fertilizer is right for your soil and crop?
                </p>
                <Button 
                  className="w-full bg-gradient-to-r from-primary to-primary/80 hover:from-primary/90 hover:to-primary/70"
                  onClick={() => navigate("/")}
                >
                  Get AI Recommendation
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FertilizerDetail;
