"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
    Loader2,
    TrendingUp,
    Map as MapIcon,
    PieChart as PieChartIcon,
    ArrowLeft
} from "lucide-react"

import {
    AreaChart,
    Area,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    BarChart,
    Bar,
    Legend,
    PieChart,
    Pie,
    Cell
} from "recharts"

// Colors for charts
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d', '#ffc658'];

export default function ExplorePage() {
    const [loading, setLoading] = useState(true)
    const [regionData, setRegionData] = useState<any[]>([])
    const [stateData, setStateData] = useState<any[]>([])
    const [yearFilter, setYearFilter] = useState("all")

    // Fetch Data
    useEffect(() => {
        async function fetchData() {
            try {
                const [regionRes, stateRes] = await Promise.all([
                    fetch("http://127.0.0.1:8000/stats/region"),
                    fetch("http://127.0.0.1:8000/stats/state")
                ])

                const rData = await regionRes.json()
                const sData = await stateRes.json()

                setRegionData(rData)
                setStateData(sData)
            } catch (err) {
                console.error("Failed to fetch stats", err)
            } finally {
                setLoading(false)
            }
        }
        fetchData()
    }, [])

    // Process Data for Charts
    // 1. Region Trends Over Time (Sum loan_amnt by year and region)
    // We need to pivot the data for Recharts AreaChart (year as X, regions as keys)
    const processRegionTrends = () => {
        const years = Array.from(new Set(regionData.map((d: any) => d.year))).sort()
        const regions = Array.from(new Set(regionData.map((d: any) => d.region)))

        return years.map(year => {
            const entry: any = { year }
            regions.forEach(region => {
                const match = regionData.find((d: any) => d.year === year && d.region === region)
                entry[region] = match ? match.loan_amnt : 0
            })
            return entry
        })
    }

    // 2. State Distribution (Top 10 states by total loan volume)
    const processTopStates = () => {
        // Group by state
        const stateTotals: Record<string, number> = {}
        stateData.forEach((d: any) => {
            if (yearFilter !== "all" && d.year.toString() !== yearFilter) return

            const val = d.loan_amnt
            stateTotals[d.addr_state] = (stateTotals[d.addr_state] || 0) + val
        })

        return Object.entries(stateTotals)
            .map(([name, value]) => ({ name, value }))
            .sort((a, b) => b.value - a.value)
            .slice(0, 15) // Top 15
    }

    if (loading) {
        return (
            <div className="min-h-screen bg-slate-950 flex items-center justify-center">
                <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
            </div>
        )
    }

    const trendData = processRegionTrends()
    const topStates = processTopStates()
    const availableYears = Array.from(new Set(stateData.map((d: any) => d.year))).sort()

    return (
        <div className="min-h-screen bg-slate-950 text-slate-100 pb-20">
            {/* Header */}
            <div className="bg-slate-900 border-b border-slate-800">
                <div className="container mx-auto px-4 py-6">
                    <Link href="/" className="text-slate-400 hover:text-white flex items-center mb-4 transition-colors">
                        <ArrowLeft className="w-4 h-4 mr-2" /> Back to Home
                    </Link>
                    <div className="flex justify-between items-end">
                        <div>
                            <h1 className="text-3xl font-bold text-white mb-2">Market Insights</h1>
                            <p className="text-slate-400">Deep dive into LendingClub historical trends (2007-2015)</p>
                        </div>
                        <div className="flex items-center gap-2">
                            {/* Year Filter for State Chart */}
                            <Select value={yearFilter} onValueChange={setYearFilter}>
                                <SelectTrigger className="w-[180px] bg-slate-800 border-slate-700 text-white">
                                    <SelectValue placeholder="Filter by Year" />
                                </SelectTrigger>
                                <SelectContent className="bg-slate-800 border-slate-700 text-white">
                                    <SelectItem value="all">All Years</SelectItem>
                                    {availableYears.map((y: any) => (
                                        <SelectItem key={y} value={y.toString()}>{y}</SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>
                        </div>
                    </div>
                </div>
            </div>

            <div className="container mx-auto px-4 py-8 space-y-8">

                {/* KPI Cards (Simplified) */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <Card className="bg-slate-900/50 border-slate-800 text-white backdrop-blur">
                        <CardHeader className="pb-2">
                            <CardTitle className="text-sm font-medium text-slate-400">Total Loan Volume Tracked</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold">
                                ${(trendData.reduce((acc, curr) => acc + Object.values(curr).slice(1).reduce((a: any, b: any) => a + b, 0), 0) / 1000000000).toFixed(2)} Billion
                            </div>
                        </CardContent>
                    </Card>

                    <Card className="bg-slate-900/50 border-slate-800 text-white backdrop-blur">
                        <CardHeader className="pb-2">
                            <CardTitle className="text-sm font-medium text-slate-400">Data Points</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold">
                                {(stateData.reduce((acc, curr) => acc + 1, 0)).toLocaleString()} Rows
                            </div>
                        </CardContent>
                    </Card>

                    <Card className="bg-slate-900/50 border-slate-800 text-white backdrop-blur">
                        <CardHeader className="pb-2">
                            <CardTitle className="text-sm font-medium text-slate-400">Top State</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold text-emerald-400">
                                {topStates[0]?.name}
                            </div>
                        </CardContent>
                    </Card>
                </div>

                {/* Chart 1: Regional Trends */}
                <Card className="bg-slate-900 border-slate-800 text-white">
                    <CardHeader>
                        <div className="flex items-center gap-2">
                            <TrendingUp className="w-5 h-5 text-blue-400" />
                            <CardTitle>Regional Growth Over Time</CardTitle>
                        </div>
                        <CardDescription className="text-slate-400">Yearly loan volume by US Region</CardDescription>
                    </CardHeader>
                    <CardContent>
                        <div className="h-[400px] w-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={trendData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                                    <defs>
                                        <linearGradient id="colorWest" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#8884d8" stopOpacity={0.8} />
                                            <stop offset="95%" stopColor="#8884d8" stopOpacity={0} />
                                        </linearGradient>
                                        <linearGradient id="colorEast" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#82ca9d" stopOpacity={0.8} />
                                            <stop offset="95%" stopColor="#82ca9d" stopOpacity={0} />
                                        </linearGradient>
                                        <linearGradient id="colorMid" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#ffc658" stopOpacity={0.8} />
                                            <stop offset="95%" stopColor="#ffc658" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <XAxis dataKey="year" stroke="#94a3b8" />
                                    <YAxis
                                        stroke="#94a3b8"
                                        tickFormatter={(value) => `$${value / 1000000}M`}
                                    />
                                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f1f5f9' }}
                                        formatter={(value: number) => [`$${(value / 1000000).toFixed(1)}M`, 'Volume']}
                                    />
                                    <Legend />
                                    <Area type="monotone" dataKey="West" stackId="1" stroke="#8884d8" fill="url(#colorWest)" />
                                    <Area type="monotone" dataKey="NorthEast" stackId="1" stroke="#82ca9d" fill="url(#colorEast)" />
                                    <Area type="monotone" dataKey="SouthEast" stackId="1" stroke="#ffc658" fill="url(#colorMid)" />
                                    <Area type="monotone" dataKey="MidWest" stackId="1" stroke="#ff8042" fill="#ff8042" />
                                    <Area type="monotone" dataKey="SouthWest" stackId="1" stroke="#0088FE" fill="#0088FE" />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </CardContent>
                </Card>

                {/* Chart 2: Top States */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    <Card className="bg-slate-900 border-slate-800 text-white">
                        <CardHeader>
                            <div className="flex items-center gap-2">
                                <MapIcon className="w-5 h-5 text-emerald-400" />
                                <CardTitle>Top 15 States by Volume</CardTitle>
                            </div>
                            <CardDescription className="text-slate-400">High volume lending markets {yearFilter !== 'all' ? `in ${yearFilter}` : '(All Time)'}</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <div className="h-[400px] w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={topStates} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
                                        <XAxis type="number" hide />
                                        <YAxis dataKey="name" type="category" width={40} stroke="#94a3b8" />
                                        <Tooltip
                                            cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                                            contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f1f5f9' }}
                                            formatter={(value: number) => [`$${(value / 1000000).toFixed(1)}M`, 'Volume']}
                                        />
                                        <Bar dataKey="value" fill="#3b82f6" radius={[0, 4, 4, 0]} barSize={20} />
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </CardContent>
                    </Card>

                    <Card className="bg-slate-900 border-slate-800 text-white">
                        <CardHeader>
                            <div className="flex items-center gap-2">
                                <PieChartIcon className="w-5 h-5 text-purple-400" />
                                <CardTitle>Volume Distribution</CardTitle>
                            </div>
                            <CardDescription className="text-slate-400">Concentration of lending across top states</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <div className="h-[400px] w-full flex items-center justify-center">
                                <ResponsiveContainer width="100%" height="100%">
                                    <PieChart>
                                        <Pie
                                            data={topStates.slice(0, 5)} // Only top 5 for pie
                                            cx="50%"
                                            cy="50%"
                                            innerRadius={60}
                                            outerRadius={100}
                                            fill="#8884d8"
                                            paddingAngle={5}
                                            dataKey="value"
                                        >
                                            {topStates.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                            ))}
                                        </Pie>
                                        <Tooltip
                                            contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f1f5f9' }}
                                            formatter={(value: number) => [`$${(value / 1000000).toFixed(1)}M`, 'Volume']}
                                        />
                                        <Legend />
                                    </PieChart>
                                </ResponsiveContainer>
                            </div>
                        </CardContent>
                    </Card>
                </div>

            </div>
        </div>
    )
}
