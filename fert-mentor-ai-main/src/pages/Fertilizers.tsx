import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Sprout, ArrowRight, Leaf, Droplets, Zap, Shield, Sun } from "lucide-react";

const fertilizers = [
  {
    id: "10-26-26",
    name: "10-26-26",
    formula: "NPK 10-26-26",
    nitrogenContent: "10% N, 26% P₂O₅, 26% K₂O",
    description: "High P and K complex fertilizer for basal dose, strong roots, flowering and fruiting.",
    icon: Zap,
    color: "primary",
    suitableFor: ["Cotton", "Groundnut", "Soybean", "Chilli", "Potato", "Vegetables", "Wheat"],
    benefits: [
      "Strong root development",
      "Enhances flowering and fruiting",
      "Improves quality and yield",
      "Supports energy metabolism",
      "Increases stress tolerance"
    ]
  },
  {
    id: "14-35-14",
    name: "14-35-14",
    formula: "NPK 14-35-14",
    nitrogenContent: "14% N, 35% P₂O₅, 14% K₂O",
    description: "High P complex with 63% total nutrients; excellent for phosphorus-demanding crops.",
    icon: Droplets,
    color: "secondary",
    suitableFor: ["Cotton", "Groundnut", "Chilli", "Soybean", "Potato", "Pulses", "Vegetables"],
    benefits: [
      "Highest phosphorus among complexes",
      "Vigorous root establishment",
      "Enhanced flowering and fruiting",
      "Better stress and disease tolerance",
      "Supports DNA/RNA formation"
    ]
  },
  {
    id: "17-17-17",
    name: "17-17-17",
    formula: "NPK 17-17-17",
    nitrogenContent: "17% N, 17% P₂O₅, 17% K₂O",
    description: "Balanced NPK for universal use across growth stages and soil types.",
    icon: Shield,
    color: "accent",
    suitableFor: ["Pomegranate", "Orange", "Oil seeds", "Vegetables", "All crops"],
    benefits: [
      "Balanced nutrition for all stages",
      "Strong roots and leafy growth",
      "Better flowering, fruiting and quality",
      "Improves drought and disease resistance"
    ]
  },
  {
    id: "28-28",
    name: "28-28",
    formula: "NP 28-28",
    nitrogenContent: "28% N, 28% P₂O₅, 0% K₂O",
    description: "High-analysis water-soluble NP for rapid vegetative growth; no potassium.",
    icon: Zap,
    color: "primary",
    suitableFor: ["Pomegranate", "Mango", "Orange", "Oil seeds", "Cereals", "Vegetables"],
    benefits: [
      "Quick vegetative growth",
      "Vigorous branching and foliage",
      "Instant nutrition in peak growth",
      "Excellent for foliar and fertigation"
    ]
  },
  {
    id: "dap",
    name: "DAP (Diammonium Phosphate)",
    formula: "(NH₄)₂HPO₄",
    nitrogenContent: "18% N, 46% P₂O₅",
    description: "Widely used P fertilizer; excellent starter for root development and legumes.",
    icon: Droplets,
    color: "secondary",
    suitableFor: ["Pulses", "Mango", "Grapes", "Apple", "Root development"],
    benefits: [
      "Readily available phosphorus",
      "Enhances root and early growth",
      "Improves nitrogen fixation",
      "Quick dissolution and uptake"
    ]
  },
  {
    id: "urea",
    name: "Urea",
    formula: "CO(NH₂)₂",
    nitrogenContent: "46% N",
    description: "Most concentrated solid nitrogen fertilizer; universal vegetative growth booster.",
    icon: Leaf,
    color: "primary",
    suitableFor: ["Paddy", "Maize", "Cotton", "Wheat", "Sugarcane", "Vegetables"],
    benefits: [
      "Highest nitrogen concentration",
      "Promotes vigorous vegetative growth",
      "Cost-effective and versatile",
      "Essential for protein and chlorophyll"
    ]
  }
];

const Fertilizers = () => {
  const navigate = useNavigate();

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
            <Button variant="outline" onClick={() => navigate("/")}>
              Get Recommendation
            </Button>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="w-full px-4 py-12">
        <div className="text-center max-w-3xl mx-auto mb-12">
          <h2 className="text-4xl md:text-5xl lg:text-6xl xl:text-7xl font-bold mb-4 bg-gradient-to-r from-primary via-primary to-secondary bg-clip-text text-transparent">
            Explore Fertilizers
          </h2>
          <p className="text-lg md:text-xl lg:text-2xl text-muted-foreground">
            Learn about different types of fertilizers and find the perfect match for your crops
          </p>
        </div>

        {/* Fertilizer Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 w-full">
          {fertilizers.map((fertilizer) => {
            const Icon = fertilizer.icon;
            return (
              <Card
                key={fertilizer.id}
                className="group hover:shadow-[var(--shadow-card)] transition-all duration-300 hover:-translate-y-1 cursor-pointer border-border/50 hover:border-primary/30"
                onClick={() => navigate(`/fertilizers/${fertilizer.id}`)}
              >
                <CardHeader>
                  <div className="flex items-start justify-between mb-4">
                    <div className={`p-3 rounded-lg bg-${fertilizer.color}/10`}>
                      <Icon className={`h-8 w-8 text-${fertilizer.color}`} />
                    </div>
                    <Badge variant="outline" className="text-xs">
                      {fertilizer.nitrogenContent}
                    </Badge>
                  </div>
                  <CardTitle className="text-xl md:text-2xl lg:text-3xl group-hover:text-primary transition-[var(--transition-smooth)]">
                    {fertilizer.name}
                  </CardTitle>
                  <CardDescription className="font-mono text-xs md:text-sm lg:text-base">
                    {fertilizer.formula}
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <p className="text-sm md:text-base lg:text-lg text-muted-foreground">
                    {fertilizer.description}
                  </p>
                  
                  <div className="space-y-2">
                    <p className="text-xs md:text-sm font-semibold text-foreground">Suitable for:</p>
                    <div className="flex flex-wrap gap-1">
                      {fertilizer.suitableFor.map((crop) => (
                        <Badge key={crop} variant="secondary" className="text-xs">
                          {crop}
                        </Badge>
                      ))}
                    </div>
                  </div>

                  <div className="space-y-2">
                    <p className="text-xs md:text-sm font-semibold text-foreground">Key Benefits:</p>
                    <ul className="space-y-1">
                      {fertilizer.benefits.map((benefit, idx) => (
                        <li key={idx} className="text-xs md:text-sm text-muted-foreground flex items-center gap-2">
                          <div className="h-1 w-1 rounded-full bg-primary"></div>
                          {benefit}
                        </li>
                      ))}
                    </ul>
                  </div>

                  <Button 
                    variant="ghost" 
                    className="w-full group-hover:bg-primary/10 group-hover:text-primary"
                  >
                    Learn More
                    <ArrowRight className="ml-2 h-4 w-4 group-hover:translate-x-1 transition-[var(--transition-smooth)]" />
                  </Button>
                </CardContent>
              </Card>
            );
          })}
        </div>

        {/* Info Section */}
        <div className="mt-16 w-full">
          <Card className="bg-gradient-to-br from-primary/5 to-secondary/5 border-primary/20">
            <CardHeader>
              <CardTitle className="text-2xl md:text-3xl lg:text-4xl">Need a Recommendation?</CardTitle>
              <CardDescription className="text-base md:text-lg lg:text-xl">
                Not sure which fertilizer to use? Our AI-powered system can analyze your soil and crop data to provide personalized recommendations.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button 
                size="lg" 
                onClick={() => navigate("/")}
                className="bg-gradient-to-r from-primary to-primary/80 hover:from-primary/90 hover:to-primary/70 shadow-[var(--shadow-soft)]"
              >
                Get AI Recommendation
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
            </CardContent>
          </Card>
        </div>
      </section>
    </div>
  );
};

export default Fertilizers;
