import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  Sprout, 
  ArrowRight, 
  Database, 
  Brain, 
  Code, 
  BarChart3, 
  CheckCircle, 
  AlertTriangle,
  Lightbulb,
  Target,
  Zap,
  Shield,
  Droplets,
  Thermometer,
  Cloud,
  Leaf,
  Settings,
  TrendingUp,
  Users,
  Globe,
  Download
} from "lucide-react";

const Reports = () => {
  const navigate = useNavigate();

  const handleDownload = (filename: string) => {
    const link = document.createElement('a');
    link.href = `/${filename}`;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const problemStatement = {
    title: "AI-Powered Fertilizer Recommendation",
    description: "Addresses the problem of excessive or improper fertilizer use, which harms soil health and increases costs for farmers. The system analyzes soil characteristics, crop type, and environmental conditions to provide precise fertilizer recommendations.",
    impact: [
      "Reduces fertilizer waste and costs",
      "Improves soil health and crop yield", 
      "Minimizes environmental impact",
      "Supports sustainable farming practices"
    ]
  };

  const dataJourney = [
    {
      step: 1,
      title: "Dataset Research & Collection",
      description: "No single dataset contained all required features. Found two complementary datasets for merging.",
      challenge: "Missing comprehensive dataset with all required features",
      solution: "Identified Crop_recommendation.csv (2200 rows) and Fertilizer Prediction.csv (96 rows)",
      icon: Database,
      color: "primary",
      details: {
        datasets: [
          { name: "Crop_recommendation.csv", rows: 2200, features: ["Temperature", "Humidity", "Moisture", "Soil_Type", "Crop_Type", "Nitrogen", "Potassium", "Phosphorous", "Fertilizer"] },
          { name: "Fertilizer Prediction.csv", rows: 96, features: ["Temperature", "Humidity", "Moisture", "Soil Type", "Crop Type", "Nitrogen", "Potassium", "Phosphorous", "Fertilizer Name"] }
        ],
        missingFeatures: ["pH", "rainfall"]
      }
    },
    {
      step: 2,
      title: "Data Standardization & Merging",
      description: "Standardized column names and merged datasets using crop-based matching.",
      challenge: "Different column naming conventions and data formats",
      solution: "Renamed columns and created unified merged_crop_fertilizer.csv",
      icon: Settings,
      color: "secondary",
      details: {
        columnMapping: {
          "Temparature": "temperature",
          "Humidity ": "humidity", 
          "Soil Type": "soil_type",
          "Crop Type": "crop",
          "Nitrogen": "N",
          "Phosphorous": "P",
          "Potassium": "K",
          "Fertilizer Name": "fertilizer_name"
        },
        mergingStrategy: "Crop-based matching between datasets"
      }
    },
    {
      step: 3,
      title: "Feature Engineering - Fertilizer Name",
      description: "Used Random Forest Classifier to predict missing fertilizer names based on NPK values.",
      challenge: "Missing fertilizer_name values in merged dataset",
      solution: "Trained RF Classifier (300 estimators) using N, P, K features only",
      icon: Brain,
      color: "accent",
      details: {
        model: "Random Forest Classifier",
        parameters: "n_estimators=300, random_state=42",
        features: ["N", "P", "K"],
        target: "fertilizer_name",
        approach: "Used only rows with existing fertilizer_name for training, then predicted missing values"
      }
    },
    {
      step: 4,
      title: "Feature Engineering - Moisture",
      description: "Used Random Forest Regressor to predict missing moisture values.",
      challenge: "Missing moisture data affecting model training",
      solution: "Trained RF Regressor using N, P, humidity, temperature features",
      icon: Droplets,
      color: "primary",
      details: {
        model: "Random Forest Regressor", 
        parameters: "n_estimators=300, random_state=42",
        features: ["N", "P", "humidity", "temperature"],
        target: "moisture",
        approach: "Used complete fertilizer dataset for training, then predicted missing values in merged dataset"
      }
    },
    {
      step: 5,
      title: "Feature Analysis - Soil Type",
      description: "Analyzed soil_type relevance and found it irrelevant for fertilizer prediction.",
      challenge: "Uncertainty about soil_type feature importance",
      solution: "Found soil_type features irrelevant, dropped entirely from final dataset",
      icon: AlertTriangle,
      color: "destructive",
      details: {
        model: "Random Forest Classifier",
        features: ["crop", "moisture", "N", "P"],
        target: "soil_type",
        decision: "Dropped soil_type - found to be irrelevant for fertilizer prediction",
        reasoning: "Low feature importance and poor correlation with fertilizer recommendations"
      }
    },
    {
      step: 6,
      title: "Data Cleaning & Finalization",
      description: "Removed irrelevant data and created final training dataset with 2200 rows.",
      challenge: "Last 100 rows contained irrelevant crop types and poor quality data",
      solution: "Dropped last 100 rows, created final dataset.csv (2200 rows)",
      icon: CheckCircle,
      color: "green",
      details: {
        finalDataset: "dataset.csv",
        finalRows: 2200,
        removedRows: 100,
        reason: "Irrelevant crop types and data quality issues",
        finalFeatures: ["N", "P", "K", "temperature", "humidity", "pH", "rainfall", "moisture", "crop", "fertilizer_name"]
      }
    }
  ];

  const models = [
    { name: "Decision Tree", train: 1.0000, val: 0.9727, test: 0.9758, color: "bg-blue-500" },
    { name: "Logistic Regression", train: 0.7896, val: 0.7485, test: 0.7485, color: "bg-red-500" },
    { name: "Random Forest", train: 1.0000, val: 0.9758, test: 0.9848, color: "bg-green-500" },
    { name: "Gradient Boosting", train: 1.0000, val: 0.9758, test: 0.9879, color: "bg-yellow-500" },
    { name: "XGBoost", train: 1.0000, val: 0.9879, test: 0.9909, color: "bg-purple-500" },
    { name: "CatBoost", train: 1.0000, val: 0.9909, test: 0.9879, color: "bg-indigo-500" }
  ];

  const features = [
    { name: "N (Nitrogen)", importance: 0.25, description: "Essential for vegetative growth" },
    { name: "P (Phosphorus)", importance: 0.22, description: "Root development and flowering" },
    { name: "K (Potassium)", importance: 0.18, description: "Fruit quality and stress resistance" },
    { name: "Temperature", importance: 0.15, description: "Crop growth rate and metabolism" },
    { name: "Humidity", importance: 0.12, description: "Water stress and disease risk" },
    { name: "pH", importance: 0.08, description: "Nutrient availability" }
  ];

  const preprocessingSteps = [
    {
      step: "Label Encoding",
      description: "Convert categorical variables to numerical format",
      details: {
        crop: "LabelEncoder() - 16 unique crop types encoded as 0-15",
        fertilizer_name: "LabelEncoder() - 6 fertilizer types encoded as 0-5"
      }
    },
    {
      step: "Standard Scaling", 
      description: "Normalize numerical features to same scale",
      details: {
        features: ["N", "P", "K", "temperature", "humidity", "pH", "rainfall", "moisture"],
        method: "StandardScaler() - mean=0, std=1",
        reason: "Ensures all features contribute equally to model training"
      }
    },
    {
      step: "Feature Selection",
      description: "Final feature set for model training",
      details: {
        totalFeatures: 9,
        numerical: 8,
        categorical: 1,
        finalSet: ["N", "P", "K", "temperature", "humidity", "pH", "rainfall", "moisture", "crop"]
      }
    }
  ];

  const technicalStack = [
    { category: "Backend", items: ["Python", "FastAPI", "XGBoost", "Scikit-learn", "Pandas", "NumPy"] },
    { category: "Frontend", items: ["React", "TypeScript", "Tailwind CSS", "shadcn/ui", "Vite"] },
    { category: "ML Pipeline", items: ["Label Encoding", "Standard Scaling", "Feature Engineering", "Model Training"] },
    { category: "Deployment", items: ["REST API", "CORS", "Error Handling", "Model Persistence"] }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/30">
      {/* Header */}
      <header className="border-b bg-card/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2 cursor-pointer" onClick={() => navigate("/")}>
            <Sprout className="h-8 w-8 text-primary" />
            <h1 className="text-2xl font-bold text-foreground">FertiSmart</h1>
          </div>
          <div className="flex items-center gap-4">
            <Button variant="outline" onClick={() => navigate("/fertilizers")}>
              Browse Fertilizers
            </Button>
            <Button variant="outline" onClick={() => navigate("/")}>
              Get Recommendation
            </Button>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="w-full px-4 py-12">
        <div className="text-center max-w-6xl mx-auto mb-12">
          <h2 className="text-4xl md:text-5xl lg:text-6xl xl:text-7xl font-bold mb-4 bg-gradient-to-r from-primary via-primary to-secondary bg-clip-text text-transparent">
            Research & Development Report
          </h2>
          <p className="text-lg md:text-xl lg:text-2xl text-muted-foreground mb-6">
            Complete journey of building an AI-powered fertilizer recommendation system
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <Badge variant="outline" className="text-sm px-4 py-2">
              <Brain className="h-4 w-4 mr-2" />
              Machine Learning
            </Badge>
            <Badge variant="outline" className="text-sm px-4 py-2">
              <Code className="h-4 w-4 mr-2" />
              Full Stack Development
            </Badge>
            <Badge variant="outline" className="text-sm px-4 py-2">
              <BarChart3 className="h-4 w-4 mr-2" />
              99.09% Accuracy
            </Badge>
          </div>
        </div>

        {/* Problem Statement */}
        <Card className="w-full mb-12 border-primary/20 bg-gradient-to-br from-primary/5 to-secondary/5">
          <CardHeader>
            <CardTitle className="text-2xl md:text-3xl lg:text-4xl flex items-center gap-2">
              <Target className="h-6 w-6 text-primary" />
              Problem Statement
            </CardTitle>
            <CardDescription className="text-base md:text-lg lg:text-xl">
              {problemStatement.description}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-semibold mb-3 text-foreground text-base md:text-lg">Key Challenges:</h4>
                <ul className="space-y-2">
                  <li className="flex items-center gap-2 text-sm md:text-base text-muted-foreground">
                    <AlertTriangle className="h-4 w-4 text-orange-500" />
                    Excessive fertilizer use harming soil health
                  </li>
                  <li className="flex items-center gap-2 text-sm md:text-base text-muted-foreground">
                    <AlertTriangle className="h-4 w-4 text-orange-500" />
                    High input costs for farmers
                  </li>
                  <li className="flex items-center gap-2 text-sm md:text-base text-muted-foreground">
                    <AlertTriangle className="h-4 w-4 text-orange-500" />
                    Environmental impact from over-fertilization
                  </li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold mb-3 text-foreground text-base md:text-lg">Our Solution Impact:</h4>
                <ul className="space-y-2">
                  {problemStatement.impact.map((impact, idx) => (
                    <li key={idx} className="flex items-center gap-2 text-sm md:text-base text-muted-foreground">
                      <CheckCircle className="h-4 w-4 text-green-500" />
                      {impact}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Main Content Tabs */}
        <Tabs defaultValue="data-journey" className="w-full">
           <TabsList className="grid w-full grid-cols-4 mb-8 h-12">
            <TabsTrigger value="data-journey">Data Journey</TabsTrigger>
            <TabsTrigger value="model-training">Model Training</TabsTrigger>
            <TabsTrigger value="system-architecture">System Design</TabsTrigger>
            <TabsTrigger value="results">Results & Impact</TabsTrigger>
          </TabsList>

          {/* Data Journey Tab */}
          <TabsContent value="data-journey" className="space-y-8">
            <Card>
              <CardHeader>
                <CardTitle className="text-2xl flex items-center gap-2">
                  <Database className="h-6 w-6 text-primary" />
                  Data Collection & Research Journey
                </CardTitle>
                <CardDescription>
                  Overview of our data engineering process from research to implementation
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-4">
                  {dataJourney.map((step, idx) => {
                    const Icon = step.icon;
                    return (
                      <div key={idx} className="relative group">
                        <Card className="h-full hover:shadow-md transition-all duration-200 border-border/50">
                          <CardHeader className="pb-3">
                            <div className="flex items-center gap-3">
                              <div className="w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-sm font-bold">
                                {step.step}
                              </div>
                              <div className={`p-2 rounded-lg bg-${step.color}/10`}>
                                <Icon className={`h-4 w-4 text-${step.color}`} />
                              </div>
                              <div className="flex-1">
                                <CardTitle className="text-base font-semibold">{step.title}</CardTitle>
                              </div>
                            </div>
                          </CardHeader>
                          <CardContent className="pt-0">
                            <p className="text-sm text-muted-foreground mb-3">{step.description}</p>
                            <div className="bg-green-50 dark:bg-green-950 p-3 rounded-lg border border-green-200 dark:border-green-800 mb-3">
                              <p className="text-green-700 dark:text-green-300 text-xs">
                                <strong>Solution:</strong> {step.solution}
                              </p>
                            </div>
                            
                            {/* Download buttons for specific steps */}
                            {step.step === 1 && (
                              <div>
                                <h4 className="text-xs font-semibold text-foreground mb-2">Download Source Datasets:</h4>
                                <div className="grid grid-cols-2 gap-2">
                                   <Button 
                                     variant="outline" 
                                     onClick={() => handleDownload("Crop_recommendation.csv")}
                                     className="flex items-center gap-2 h-10 px-4"
                                     size="sm"
                                   >
                                     <Download className="h-4 w-4" />
                                     <div className="text-left">
                                       <div className="text-sm font-medium">Crop_recommendation.csv (2200 rows)</div>
                                     </div>
                                   </Button>
                                   <Button 
                                     variant="outline" 
                                     onClick={() => handleDownload("Fertilizer Prediction.csv")}
                                     className="flex items-center gap-2 h-10 px-4"
                                     size="sm"
                                   >
                                     <Download className="h-4 w-4" />
                                     <div className="text-left">
                                       <div className="text-sm font-medium">Fertilizer Prediction.csv (96 rows)</div>
                                     </div>
                                   </Button>
                                </div>
                              </div>
                            )}
                            
                            {step.step === 6 && (
                              <div>
                                <h4 className="text-xs font-semibold text-foreground mb-2">Download Final Dataset:</h4>
                                 <Button 
                                   variant="outline" 
                                   onClick={() => handleDownload("dataset.csv")}
                                   className="flex items-center gap-2 h-10 px-4"
                                   size="sm"
                                 >
                                   <Download className="h-4 w-4" />
                                   <div className="text-left">
                                     <div className="text-sm font-medium">dataset.csv (2200 rows - Clean & Processed)</div>
                                   </div>
                                 </Button>
                              </div>
                            )}
                          </CardContent>
                        </Card>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>


            <Card>
              <CardHeader>
                <CardTitle className="text-xl flex items-center gap-2">
                  <Settings className="h-6 w-6 text-primary" />
                  Data Preprocessing Pipeline
                </CardTitle>
                <CardDescription>Final preprocessing steps before model training</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  {preprocessingSteps.map((step, idx) => (
                    <div key={idx} className="flex gap-4 p-4 rounded-lg bg-muted/30">
                      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
                        <span className="text-sm font-semibold text-primary">{idx + 1}</span>
                      </div>
                      <div className="flex-1">
                        <h4 className="font-semibold text-foreground mb-2">{step.step}</h4>
                        <p className="text-sm text-muted-foreground mb-3">{step.description}</p>
                        <div className="grid md:grid-cols-2 gap-4">
                          {Object.entries(step.details).map(([key, value]) => (
                            <div key={key} className="space-y-1">
                              <h5 className="text-xs font-medium text-foreground uppercase tracking-wide">
                                {key.replace(/([A-Z])/g, ' $1').trim()}
                              </h5>
                              {Array.isArray(value) ? (
                                <div className="flex flex-wrap gap-1">
                                  {value.map((item, i) => (
                                    <Badge key={i} variant="secondary" className="text-xs">
                                      {item}
                                    </Badge>
                                  ))}
                                </div>
                              ) : (
                                <p className="text-sm text-muted-foreground">{value}</p>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Model Training Tab */}
          <TabsContent value="model-training" className="space-y-8">
            <Card>
              <CardHeader>
                <CardTitle className="text-2xl flex items-center gap-2">
                  <Brain className="h-6 w-6 text-primary" />
                  Model Training & Evaluation
                </CardTitle>
                <CardDescription>
                  Comprehensive evaluation of 6 different machine learning algorithms
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  <div className="grid gap-4">
                    <h4 className="font-semibold text-lg">Model Performance Comparison</h4>
                    <div className="overflow-x-auto">
                      <table className="w-full border-collapse min-w-full">
                        <thead>
                          <tr className="border-b">
                            <th className="text-left p-3 font-medium">Model</th>
                            <th className="text-center p-3 font-medium">Train Accuracy</th>
                            <th className="text-center p-3 font-medium">Validation Accuracy</th>
                            <th className="text-center p-3 font-medium">Test Accuracy</th>
                          </tr>
                        </thead>
                        <tbody>
                          {models.map((model, idx) => (
                            <tr key={idx} className="border-b hover:bg-muted/30">
                              <td className="p-3 font-medium">{model.name}</td>
                              <td className="text-center p-3">{(model.train * 100).toFixed(2)}%</td>
                              <td className="text-center p-3">{(model.val * 100).toFixed(2)}%</td>
                              <td className="text-center p-3 font-semibold">{(model.test * 100).toFixed(2)}%</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  <div className="p-6 rounded-lg bg-gradient-to-r from-purple-50 to-blue-50 border border-purple-200">
                    <div className="flex items-center gap-3 mb-3">
                      <Zap className="h-6 w-6 text-purple-600" />
                      <h4 className="text-lg font-semibold text-purple-800">Selected Model: XGBoost</h4>
                    </div>
                    <p className="text-purple-700 mb-4">
                      XGBoost achieved the highest test accuracy of <strong>99.09%</strong> with excellent 
                      performance across all metrics. It showed the best balance of accuracy, 
                      generalization, and computational efficiency.
                    </p>
                    <div className="grid md:grid-cols-3 gap-4">
                      <div className="text-center">
                        <div className="text-2xl font-bold text-purple-600">99.09%</div>
                        <div className="text-sm text-purple-700">Test Accuracy</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-purple-600">98.79%</div>
                        <div className="text-sm text-purple-700">Validation Accuracy</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-purple-600">100%</div>
                        <div className="text-sm text-purple-700">Training Accuracy</div>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h4 className="font-semibold text-lg mb-4">Feature Importance Analysis</h4>
                    <div className="space-y-3">
                      {features.map((feature, idx) => (
                        <div key={idx} className="flex items-center gap-4">
                          <div className="w-32 text-sm font-medium">{feature.name}</div>
                          <div className="flex-1 bg-muted rounded-full h-2">
                            <div 
                              className="bg-primary h-2 rounded-full transition-all duration-500"
                              style={{ width: `${feature.importance * 100}%` }}
                            />
                          </div>
                          <div className="w-16 text-sm text-muted-foreground text-right">
                            {(feature.importance * 100).toFixed(0)}%
                          </div>
                        </div>
                      ))}
                    </div>
                    <p className="text-sm text-muted-foreground mt-3">
                      Nitrogen (N), Phosphorus (P), and Potassium (K) are the most important features, 
                      which aligns with agricultural science principles.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* System Architecture Tab */}
          <TabsContent value="system-architecture" className="space-y-8">
            <Card>
              <CardHeader>
                <CardTitle className="text-2xl flex items-center gap-2">
                  <Settings className="h-6 w-6 text-primary" />
                  System Architecture
                </CardTitle>
                <CardDescription>
                  Complete technical stack and system design
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    {technicalStack.map((category, idx) => (
                      <div key={idx} className="p-4 rounded-lg bg-muted/30">
                        <h4 className="font-semibold text-lg mb-3 text-foreground">{category.category}</h4>
                        <div className="flex flex-wrap gap-2">
                          {category.items.map((item, itemIdx) => (
                            <Badge key={itemIdx} variant="secondary" className="text-xs">
                              {item}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="p-6 rounded-lg bg-gradient-to-r from-blue-50 to-green-50 border border-blue-200">
                    <h4 className="font-semibold text-lg mb-4 text-blue-800">Data Flow Pipeline</h4>
                    <div className="flex flex-wrap items-center gap-4 text-sm">
                      <div className="flex items-center gap-2 px-3 py-2 bg-white rounded border">
                        <Database className="h-4 w-4 text-blue-600" />
                        Raw Data
                      </div>
                      <ArrowRight className="h-4 w-4 text-muted-foreground" />
                      <div className="flex items-center gap-2 px-3 py-2 bg-white rounded border">
                        <Settings className="h-4 w-4 text-green-600" />
                        Preprocessing
                      </div>
                      <ArrowRight className="h-4 w-4 text-muted-foreground" />
                      <div className="flex items-center gap-2 px-3 py-2 bg-white rounded border">
                        <Brain className="h-4 w-4 text-purple-600" />
                        Model Training
                      </div>
                      <ArrowRight className="h-4 w-4 text-muted-foreground" />
                      <div className="flex items-center gap-2 px-3 py-2 bg-white rounded border">
                        <Code className="h-4 w-4 text-orange-600" />
                        API Development
                      </div>
                      <ArrowRight className="h-4 w-4 text-muted-foreground" />
                      <div className="flex items-center gap-2 px-3 py-2 bg-white rounded border">
                        <Globe className="h-4 w-4 text-indigo-600" />
                        Web Interface
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Results & Impact Tab */}
          <TabsContent value="results" className="space-y-8">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <Card className="border-green-200 bg-gradient-to-br from-green-50 to-emerald-50">
                <CardHeader>
                  <CardTitle className="text-green-800 flex items-center gap-2">
                    <TrendingUp className="h-5 w-5" />
                    Model Performance
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-green-600 mb-2">99.09%</div>
                  <p className="text-green-700 text-sm">Test Accuracy</p>
                  <p className="text-green-600 text-xs mt-2">Industry-leading performance</p>
                </CardContent>
              </Card>

              <Card className="border-blue-200 bg-gradient-to-br from-blue-50 to-cyan-50">
                <CardHeader>
                  <CardTitle className="text-blue-800 flex items-center gap-2">
                    <Users className="h-5 w-5" />
                    User Experience
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-blue-600 mb-2">Intuitive</div>
                  <p className="text-blue-700 text-sm">Clean Interface</p>
                  <p className="text-blue-600 text-xs mt-2">Real-time predictions</p>
                </CardContent>
              </Card>

              <Card className="border-purple-200 bg-gradient-to-br from-purple-50 to-violet-50">
                <CardHeader>
                  <CardTitle className="text-purple-800 flex items-center gap-2">
                    <Shield className="h-5 w-5" />
                    Sustainability
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-purple-600 mb-2">Eco-Friendly</div>
                  <p className="text-purple-700 text-sm">Reduced Waste</p>
                  <p className="text-purple-600 text-xs mt-2">Optimized fertilizer use</p>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle className="text-2xl flex items-center gap-2">
                  <BarChart3 className="h-6 w-6 text-primary" />
                  Key Achievements
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <h4 className="font-semibold text-lg text-foreground">Technical Achievements</h4>
                    <ul className="space-y-2">
                      <li className="flex items-center gap-2 text-sm text-muted-foreground">
                        <CheckCircle className="h-4 w-4 text-green-500" />
                        Solved missing dataset problem through feature engineering
                      </li>
                      <li className="flex items-center gap-2 text-sm text-muted-foreground">
                        <CheckCircle className="h-4 w-4 text-green-500" />
                        Achieved 99.09% accuracy with XGBoost model
                      </li>
                      <li className="flex items-center gap-2 text-sm text-muted-foreground">
                        <CheckCircle className="h-4 w-4 text-green-500" />
                        Built complete full-stack application
                      </li>
                      <li className="flex items-center gap-2 text-sm text-muted-foreground">
                        <CheckCircle className="h-4 w-4 text-green-500" />
                        Implemented real-time prediction API
                      </li>
                    </ul>
                  </div>
                  <div className="space-y-4">
                    <h4 className="font-semibold text-lg text-foreground">Impact on Agriculture</h4>
                    <ul className="space-y-2">
                      <li className="flex items-center gap-2 text-sm text-muted-foreground">
                        <CheckCircle className="h-4 w-4 text-green-500" />
                        Reduces fertilizer waste and costs
                      </li>
                      <li className="flex items-center gap-2 text-sm text-muted-foreground">
                        <CheckCircle className="h-4 w-4 text-green-500" />
                        Improves soil health and crop yield
                      </li>
                      <li className="flex items-center gap-2 text-sm text-muted-foreground">
                        <CheckCircle className="h-4 w-4 text-green-500" />
                        Minimizes environmental impact
                      </li>
                      <li className="flex items-center gap-2 text-sm text-muted-foreground">
                        <CheckCircle className="h-4 w-4 text-green-500" />
                        Supports sustainable farming practices
                      </li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="border-primary/20 bg-gradient-to-br from-primary/5 to-secondary/5">
              <CardHeader>
                <CardTitle className="text-xl">Future Enhancements</CardTitle>
                <CardDescription>Planned improvements and additional features</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <h4 className="font-semibold text-foreground">Technical Improvements</h4>
                    <ul className="space-y-1 text-sm text-muted-foreground">
                      <li>• Growth stage integration</li>
                      <li>• Dosage and timing recommendations</li>
                      <li>• Sensor data integration</li>
                      <li>• Historical data analysis</li>
                    </ul>
                  </div>
                  <div className="space-y-2">
                    <h4 className="font-semibold text-foreground">User Experience</h4>
                    <ul className="space-y-1 text-sm text-muted-foreground">
                      <li>• Mobile app development</li>
                      <li>• Multi-language support</li>
                      <li>• Farmer education modules</li>
                      <li>• Cost savings calculator</li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        {/* Call to Action */}
        <div className="mt-16 w-full">
          <Card className="bg-gradient-to-br from-primary/5 to-secondary/5 border-primary/20">
            <CardHeader>
              <CardTitle className="text-2xl text-center">Ready to Experience the System?</CardTitle>
              <CardDescription className="text-center text-base">
                Try our AI-powered fertilizer recommendation system and see the results for yourself
              </CardDescription>
            </CardHeader>
            <CardContent className="text-center">
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Button 
                  size="lg" 
                  onClick={() => navigate("/")}
                  className="bg-gradient-to-r from-primary to-primary/80 hover:from-primary/90 hover:to-primary/70 shadow-[var(--shadow-soft)]"
                >
                  Get AI Recommendation
                  <ArrowRight className="ml-2 h-5 w-5" />
                </Button>
                <Button 
                  variant="outline" 
                  size="lg"
                  onClick={() => navigate("/fertilizers")}
                >
                  Explore Fertilizers
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>
    </div>
  );
};

export default Reports;
