import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Sparkles, TrendingUp, Shield } from "lucide-react"

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950">
      {/* Hero Section */}
      <div className="container mx-auto px-4 py-16">
        <div className="flex flex-col items-center justify-center min-h-[80vh] text-center space-y-8">
          <Badge variant="outline" className="backdrop-blur-sm bg-white/5 border-white/10 text-white px-4 py-2">
            <Sparkles className="w-4 h-4 mr-2 inline" />
            Powered by Advanced Machine Learning
          </Badge>

          <h1 className="text-6xl md:text-7xl font-bold bg-gradient-to-r from-white via-blue-100 to-indigo-200 bg-clip-text text-transparent">
            LendingClub
            <br />
            Loan Prediction
          </h1>

          <p className="text-xl md:text-2xl text-slate-300 max-w-2xl">
            Get instant approval odds using state-of-the-art LightGBM model.
            <br />
            <span className="text-emerald-400 font-semibold">80.25% accuracy</span> on real lending data.
          </p>

          <div className="flex gap-4 flex-wrap justify-center">
            <Link href="/predict">
              <Button size="lg" className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-lg px-8 py-6">
                Get Pre-Approved
              </Button>
            </Link>
            <Link href="/explore">
              <Button size="lg" variant="outline" className="backdrop-blur-sm bg-white/5 border-white/10 hover:bg-white/10 text-white text-lg px-8 py-6">
                Explore Data
              </Button>
            </Link>
          </div>
        </div>

        {/* Feature Cards */}
        <div className="grid md:grid-cols-3 gap-6 mt-16">
          <Card className="backdrop-blur-md bg-white/5 border-white/10 text-white">
            <CardHeader>
              <TrendingUp className="w-10 h-10 text-emerald-400 mb-2" />
              <CardTitle>Modern AI</CardTitle>
              <CardDescription className="text-slate-300">
                Upgraded from 5-year-old models to cutting-edge LightGBM
              </CardDescription>
            </CardHeader>
          </Card>

          <Card className="backdrop-blur-md bg-white/5 border-white/10 text-white">
            <CardHeader>
              <Shield className="w-10 h-10 text-blue-400 mb-2" />
              <CardTitle>Privacy First</CardTitle>
              <CardDescription className="text-slate-300">
                Your data is never stored. All predictions run in real-time.
              </CardDescription>
            </CardHeader>
          </Card>

          <Card className="backdrop-blur-md bg-white/5 border-white/10 text-white">
            <CardHeader>
              <Sparkles className="w-10 h-10 text-purple-400 mb-2" />
              <CardTitle>Instant Results</CardTitle>
              <CardDescription className="text-slate-300">
                Get your approval odds in seconds. No credit score impact.
              </CardDescription>
            </CardHeader>
          </Card>
        </div>
      </div>
    </div>
  )
}
