import React from 'react';
import { ArrowLeft, Github, Linkedin, Mail, Heart, Code, Zap } from 'lucide-react';
import MultiColorText from '../components/animations/MultiColorText';
import ProfileCard from '../ui/ProfileCard';
import AnimatedEye from '../components/animations/AnimatedEye';
import mePng from '../assets/me.png';

interface AboutPageProps {
  onBack: () => void;
}

const AboutPage: React.FC<AboutPageProps> = ({ onBack }) => {
  return (
    <div className="min-h-screen p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="flex items-center mb-12">
          <button
            onClick={onBack}
            className="flex items-center gap-2 text-gray-700 hover:text-purple-700 transition-colors group"
          >
            <ArrowLeft size={20} className="group-hover:-translate-x-1 transition-transform" />
            <span>Back to Detection</span>
          </button>
        </div>

        {/* Hero Section */}
        <div className="text-center mb-16">
          <div className="mb-6">
            <AnimatedEye size={80} color="#8B5CF6" />
          </div>
          <h1 className="text-6xl font-bold text-gray-900 mb-6">
            <MultiColorText
              text="About Truthful"
              colors={['#8B5CF6', '#EC4899', '#3B82F6']}
            />
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Your eyes in an ever-growing digital world
          </p>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-start">
          {/* Project Info */}
          <div className="space-y-8">
            <div className="bg-white rounded-2xl p-8 shadow-md border border-purple-100">
              <h2 className="text-3xl font-bold text-gray-900 mb-6 flex items-center gap-3">
                <Zap className="text-purple-600" size={32} />
                The Mission
              </h2>
              <div className="space-y-4 text-gray-600 leading-relaxed">
                <p>
                  In an era where AI-generated content & fraud is becoming increasingly sophisticated, 
                  Truthful serves as your digital guardian, helping you distinguish between 
                  authentic and artificial to see the truth.
                </p>
                <p>
                  Our advanced detection system combines cutting-edge machine learning with 
                  skeleton-based structural analysis to provide accurate, reliable results 
                  across diverse video sources and generation methods. 
                </p>
                <p>
                  Soon to be an all in one fraud detection and security platform, we believe in transparency, accuracy, 
                  and empowering users with the tools they need to navigate today's digital landscape with confidence.  
                </p>
              </div>
            </div>

            <div className="bg-white rounded-2xl p-8 shadow-md border border-purple-100">
              <h2 className="text-3xl font-bold text-gray-900 mb-6 flex items-center gap-3">
                <Code className="text-blue-600" size={32} />
                Technology Stack
              </h2>
              <div className="grid grid-cols-2 gap-4">
                {[
                  'React + TypeScript',
                  'FastAPI Backend',
                  'PyTorch ML Models',
                  'Skeleton-Based Detection',
                  'VidProM Dataset',
                  'Real-time Processing',
                  'Supabase Database',
                  'Modern UI/UX'
                ].map((tech, index) => (
                  <div key={index} className="bg-purple-50 rounded-lg p-3 text-center text-gray-700 text-sm">
                    {tech}
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-white rounded-2xl p-8 shadow-md border border-purple-100">
              <h2 className="text-3xl font-bold text-gray-900 mb-6 flex items-center gap-3">
                <Heart className="text-red-600" size={32} />
                Open Source
              </h2>
              <p className="text-gray-600 leading-relaxed mb-4">
                Truthful is currently in open source beta and all detection algorithms and 
                methodologies are open for review, ensuring trust and continuous improvement 
                through community feedback.
              </p>
              <div className="flex gap-4">
                <a
                  href="https://github.com/your-username/truthful-ai-detector"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 bg-gray-800 hover:bg-gray-700 text-white px-4 py-2 rounded-lg transition-colors"
                >
                  <Github size={16} />
                  View Source
                </a>
                <a
                  href="https://github.com/your-username/truthful-ai-detector/issues"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg transition-colors"
                >
                  <Code size={16} />
                  Contribute
                </a>
              </div>
            </div>
          </div>

          {/* Creator Profile */}
          <div className="space-y-8">
            <div className="text-center">
              <h2 className="text-3xl font-bold text-gray-900 mb-6">
                <MultiColorText
                  text="The Founder"
                  colors={['#8B5CF6', '#EC4899', '#3B82F6']}
                />
              </h2>
              <div className="flex justify-center">
                <ProfileCard
                  name="Tahmeed Tabeeb"
                  title="Developer"
                  handle="tahmeed_t"
                  status="Building the Future"
                  contactText="Get in Touch"
                  avatarUrl={mePng}
                  onContactClick={() => window.open('mailto:tabeeb700@gmail.com')}
                />
              </div>
            </div>

            <div className="bg-white rounded-2xl p-8 shadow-md border border-purple-100">
              <h3 className="text-2xl font-bold text-gray-900 mb-4">Background</h3>
              <div className="space-y-4 text-gray-600 leading-relaxed">
                <p>
                  Hi! My name is Tahmeed and I'm a Computer Science and Business Double Degree student at Laurier in Ontario Canada. 
                  As a kid who grew up with videogames like cyberpunk, AI has been an interest of mine for a while, Truthful was designed with that same interest
                  in mind while tailoring to the secuirty concerns of the everyday user highlighting how the development of AI also acccelerates the development of security.
                </p>
                <p>
                  My machine learning experience extends from projects to classwork and coop experience (looking into breaking into the AI & ML industry in the future). 
                  Truthful came to me as an idea beyond just an everyday ai checker, but an all in one security platform/application for 
                  the general user and businesses, which will be one of the most important aspects in a accelerating digital world.
                </p>
                <p>
                  In an age where the truth can be hard to find, we stand as a platform for authenticity.
                </p>
              </div>
            </div>

            <div className="bg-white rounded-2xl p-8 shadow-md border border-purple-100">
              <h3 className="text-2xl font-bold text-gray-900 mb-4">Connect</h3>
              <div className="space-y-3">
                <a
                  href="mailto:your-email@example.com"
                  className="flex items-center gap-3 text-gray-700 hover:text-purple-700 transition-colors"
                >
                  <Mail size={20} />
                  <span>tabeeb700@gmail.com</span>
                </a>
                <a
                  href="https://linkedin.com/in/tahmeedt"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-3 text-gray-700 hover:text-purple-700 transition-colors"
                >
                  <Linkedin size={20} />
                  <span>LinkedIn Profile</span>
                </a>
                <a
                  href="https://github.com/ruyot"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-3 text-gray-700 hover:text-purple-700 transition-colors"
                >
                  <Github size={20} />
                  <span>GitHub Profile</span>
                </a>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-16 pt-8 border-t border-purple-100">
          <p className="text-gray-600">
            Built with ❤️ for a more transparent digital world
          </p>
        </div>
      </div>
    </div>
  );
};

export default AboutPage;