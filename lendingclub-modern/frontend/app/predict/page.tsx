"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Loader2, CheckCircle2, XCircle, Sparkles } from "lucide-react"
import Link from "next/link"

export default function PredictPage() {
    const [formData, setFormData] = useState({
        term: " 36 months",
        loan_amnt: "",
        grade: "A",
        emp_length: "10+ years",
        home_ownership: "RENT",
        annual_inc: "",
        purpose: "debt_consolidation"
    })

    const [result, setResult] = useState<any>(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState("")
    // Animation steps state
    const [loadingStep, setLoadingStep] = useState("")

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        setLoading(true)
        setError("")
        setResult(null)
        setLoadingStep("Validating application details...")

        try {
            // Artificial delay for UX "Analysis" feel
            await new Promise(r => setTimeout(r, 800))
            setLoadingStep("Connecting to simple ML model inference...")

            await new Promise(r => setTimeout(r, 800))
            setLoadingStep("Running LightGBM risk assessment...")

            const response = await fetch("http://127.0.0.1:8000/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    ...formData,
                    loan_amnt: parseFloat(formData.loan_amnt),
                    annual_inc: parseFloat(formData.annual_inc)
                })
            })

            if (!response.ok) throw new Error("Prediction failed")

            const data = await response.json()

            await new Promise(r => setTimeout(r, 600)) // Final small delay
            setResult(data)
        } catch (err) {
            setError("Failed to get prediction. Please check your backend is running.")
        } finally {
            setLoading(false)
            setLoadingStep("")
        }
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950 py-12">
            <div className="container mx-auto px-4 max-w-4xl">
                <Link href="/" className="text-blue-400 hover:text-blue-300 mb-8 inline-block transition-transform hover:-translate-x-1">
                    ← Back to Home
                </Link>

                <div className="text-center mb-12 animate-in fade-in slide-in-from-bottom-4 duration-700">
                    <Badge variant="outline" className="backdrop-blur-sm bg-white/5 border-white/10 text-emerald-400 mb-4 px-3 py-1">
                        <Sparkles className="w-3 h-3 mr-2 inline" />
                        Powered by LightGBM
                    </Badge>
                    <h1 className="text-5xl font-bold bg-gradient-to-r from-white via-blue-100 to-indigo-200 bg-clip-text text-transparent mb-4">
                        Get Pre-Approved
                    </h1>
                    <p className="text-slate-400 text-lg max-w-2xl mx-auto">
                        Check your loan approval odds instantly with our advanced AI model.
                        <br />No credit score impact. Real-time analysis.
                    </p>
                </div>

                <Card className="backdrop-blur-xl bg-white/5 border-white/10 text-white shadow-2xl overflow-hidden relative">
                    {/* Decorative gradients */}
                    <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 via-purple-500 to-emerald-500" />

                    <CardHeader>
                        <CardTitle className="text-2xl flex items-center gap-2">
                            Loan Application
                        </CardTitle>
                        <CardDescription className="text-slate-400">
                            Fill in your details below to get instant approval odds
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <form onSubmit={handleSubmit} className="space-y-6">
                            <div className="grid md:grid-cols-2 gap-6">
                                <div className="space-y-2">
                                    <Label htmlFor="loan_amnt">Loan Amount ($)</Label>
                                    <Input
                                        id="loan_amnt"
                                        type="number"
                                        placeholder="e.g., 10000"
                                        value={formData.loan_amnt}
                                        onChange={(e) => setFormData({ ...formData, loan_amnt: e.target.value })}
                                        required
                                        className="bg-white/10 border-white/20 text-white placeholder:text-slate-400"
                                    />
                                </div>

                                <div className="space-y-2">
                                    <Label htmlFor="annual_inc">Annual Income ($)</Label>
                                    <Input
                                        id="annual_inc"
                                        type="number"
                                        placeholder="e.g., 75000"
                                        value={formData.annual_inc}
                                        onChange={(e) => setFormData({ ...formData, annual_inc: e.target.value })}
                                        required
                                        className="bg-white/10 border-white/20 text-white placeholder:text-slate-400"
                                    />
                                </div>

                                <div className="space-y-2">
                                    <Label htmlFor="term">Loan Term</Label>
                                    <Select value={formData.term} onValueChange={(val) => setFormData({ ...formData, term: val })}>
                                        <SelectTrigger className="bg-white/10 border-white/20 text-white">
                                            <SelectValue />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value=" 36 months">36 months</SelectItem>
                                            <SelectItem value=" 60 months">60 months</SelectItem>
                                        </SelectContent>
                                    </Select>
                                </div>

                                <div className="space-y-2">
                                    <Label htmlFor="grade">Credit Grade</Label>
                                    <Select value={formData.grade} onValueChange={(val) => setFormData({ ...formData, grade: val })}>
                                        <SelectTrigger className="bg-white/10 border-white/20 text-white">
                                            <SelectValue />
                                        </SelectTrigger>
                                        <SelectContent>
                                            {['A', 'B', 'C', 'D', 'E', 'F', 'G'].map(grade => (
                                                <SelectItem key={grade} value={grade}>{grade}</SelectItem>
                                            ))}
                                        </SelectContent>
                                    </Select>
                                </div>

                                <div className="space-y-2">
                                    <Label htmlFor="emp_length">Employment Length</Label>
                                    <Select value={formData.emp_length} onValueChange={(val) => setFormData({ ...formData, emp_length: val })}>
                                        <SelectTrigger className="bg-white/10 border-white/20 text-white">
                                            <SelectValue />
                                        </SelectTrigger>
                                        <SelectContent>
                                            {['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'].map(len => (
                                                <SelectItem key={len} value={len}>{len}</SelectItem>
                                            ))}
                                        </SelectContent>
                                    </Select>
                                </div>

                                <div className="space-y-2">
                                    <Label htmlFor="home_ownership">Home Ownership</Label>
                                    <Select value={formData.home_ownership} onValueChange={(val) => setFormData({ ...formData, home_ownership: val })}>
                                        <SelectTrigger className="bg-white/10 border-white/20 text-white">
                                            <SelectValue />
                                        </SelectTrigger>
                                        <SelectContent>
                                            {['RENT', 'OWN', 'MORTGAGE', 'OTHER'].map(own => (
                                                <SelectItem key={own} value={own}>{own}</SelectItem>
                                            ))}
                                        </SelectContent>
                                    </Select>
                                </div>

                                <div className="space-y-2 md:col-span-2">
                                    <Label htmlFor="purpose">Loan Purpose</Label>
                                    <Select value={formData.purpose} onValueChange={(val) => setFormData({ ...formData, purpose: val })}>
                                        <SelectTrigger className="bg-white/10 border-white/20 text-white">
                                            <SelectValue />
                                        </SelectTrigger>
                                        <SelectContent>
                                            {['debt_consolidation', 'credit_card', 'home_improvement', 'other', 'major_purchase', 'medical', 'small_business', 'car', 'vacation', 'moving', 'house', 'wedding', 'renewable_energy', 'educational'].map(purpose => (
                                                <SelectItem key={purpose} value={purpose}>
                                                    {purpose.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                                </SelectItem>
                                            ))}
                                        </SelectContent>
                                    </Select>
                                </div>
                            </div>

                            <Button
                                type="submit"
                                disabled={loading}
                                className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-lg py-6 transition-all duration-300 relative overflow-hidden"
                            >
                                {/* Background shimmer effect during loading */}
                                {loading && (
                                    <div className="absolute inset-0 bg-white/20 animate-[shimmer_2s_infinite] skew-x-12" />
                                )}

                                {loading ? (
                                    <div className="flex items-center gap-2">
                                        <Loader2 className="h-5 w-5 animate-spin" />
                                        <span>Analyzing...</span>
                                    </div>
                                ) : (
                                    "Get Approval Odds"
                                )}
                            </Button>

                            {/* Progress Text Animation */}
                            {loading && (
                                <div className="text-center text-sm text-blue-300 animate-pulse mt-2 font-medium">
                                    {loadingStep}
                                </div>
                            )}
                        </form>

                        {error && (
                            <div className="mt-6 p-4 bg-red-500/20 border border-red-500/50 rounded-lg flex items-start gap-3">
                                <XCircle className="w-5 h-5 text-red-400 mt-0.5" />
                                <div className="text-red-200">{error}</div>
                            </div>
                        )}

                        {result && (
                            <div className="mt-8 p-6 bg-gradient-to-br from-emerald-500/20 to-blue-500/20 border border-emerald-500/30 rounded-lg">
                                <div className="flex items-start gap-3 mb-4">
                                    <CheckCircle2 className="w-6 h-6 text-emerald-400 mt-1" />
                                    <div>
                                        <h3 className="text-2xl font-bold text-white mb-2">
                                            {result.approval_chance} Approval Chance
                                        </h3>
                                        <p className="text-slate-200">{result.message}</p>
                                    </div>
                                </div>
                                <div className="mt-4 p-4 bg-white/5 rounded-lg">
                                    <div className="text-sm text-slate-300">
                                        <strong className="text-white">Model:</strong> LightGBM (80.25% accuracy)
                                    </div>
                                </div>
                            </div>
                        )}
                    </CardContent>
                </Card>
            </div>
        </div>
    )
}
