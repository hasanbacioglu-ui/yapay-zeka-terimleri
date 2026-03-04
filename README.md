# yapay-zeka-terimleri
Türkçe Yapay Zeka Terimleri

# 🤖 Ajanik YZ (Agentic AI) Terimler Rehberi

*Yapay zeka ajanları dünyasının kapsamlı sözlüğü - Temel kavramlardan ileri düzey kurumsal uygulamalara*

Bu rehber, yapay zeka alanındaki en güncel terimleri ve kavramları açıklar. Geliştiriciler, ürün yöneticileri ve öğrenmek isteyenler için hazırlanmıştır. Son güncelleme: **Mart 2025**

---

## 📚 İçindekiler

- [Temel Ajan Kavramları](#temel-ajan-kavramları)
- [Model ve Çerçeve Terimleri](#model-ve-çerçeve-terimleri)
- [Geliştirme ve Dağıtım](#geliştirme-ve-dağıtım)
- [Güvenlik ve Uyumluluk](#güvenlik-ve-uyumluluk)
- [Performans ve İzleme](#performans-ve-i̇zleme)
- [Entegrasyon ve Protokoller](#entegrasyon-ve-protokoller)
- [İş ve Operasyonel Terimler](#i̇ş-ve-operasyonel-terimler)
- [İleri Düzey Kavramlar](#i̇leri-düzey-kavramlar)

---

## Temel Ajan Kavramları

### **Ajan (Agent)**
Kendi kendine karar verebilen, çevresini algılayan ve belirli hedeflere ulaşmak için eylemde bulunan özerk yapay zeka sistemi. Geleneksel yazılımların aksine, karmaşık senaryolara uyum sağlayabilir ve akıl yürütebilir.

### **Ajanik YZ (Agentic AI)**
Kullanıcı adına özerk bir şekilde hareket eden, sürekli insan müdahalesi olmadan kararlar alan ve eylemler gerçekleştiren yapay zeka sistemleri. Planlama yapabilir, görevleri yürütebilir ve değişen koşullara adapte olabilirler.

### **Çoklu Ajan Sistemi (Multi-Agent System - MAS)**
Tek bir ajanın çözemeyeceği karmaşık sorunları çözmek için birlikte çalışan, iletişim kuran ve koordine olan birden fazla yapay zeka ajanının oluşturduğu sistem.

### **Ajan Orkestrasyonu (Agent Orchestration)**
Birden fazla ajanın birlikte çalışmasının koordinasyonu ve yönetimi; görev dağılımı, iletişim protokolleri ve sonuçların birleştirilmesini içerir.

### **Araç Çağırma (Tool Calling)**
Bir yapay zeka ajanının, temel yeteneklerinin ötesinde bilgi toplamak veya eylem gerçekleştirmek için harici fonksiyonları, API'leri veya hizmetleri çağırma yeteneği.

### **Fonksiyon Çağırma (Function Calling)**
Araç çağırmanın spesifik bir uygulaması; yapay zeka modelinin belirli görevleri yerine getirmek için yapılandırılmış parametrelerle önceden tanımlanmış fonksiyonları çağırabilmesidir.

### **Akıl Yürütme (Reasoning)**
Ajanın mevcut veri ve bağlama dayanarak bilgiyi işleme, sonuçlar çıkarma ve mantıklı kararlar alma yeteneği.

### **Planlama (Planning)**
Ajanın karmaşık hedefleri daha küçük, uygulanabilir adımlara bölme ve amaçlara ulaşmak için gereken eylem dizisini belirleme süreci.

### **Bağlam Penceresi (Context Window)**
Bir yapay zeka modelinin tek bir konuşmada veya etkileşimde işleyip hatırlayabileceği metin miktarı (token cinsinden ölçülür).

### **Oturum Yönetimi (Session Management)**
Bir ajanla yapılan birden fazla etkileşim arasında kalıcı konuşmaların ve bağlamın yönetimi; istekler arasında durum ve belleği koruma.

### **Ajan İş Akışı (Agent Workflow)**
Bir ajanın karmaşık hedefleri gerçekleştirmek için izlediği yapılandırılmış görev ve karar noktaları dizisi; genellikle birden fazla araç ve akıl yürütme adımı içerir.

### **Özerk Karar Verme (Autonomous Decision Making)**
Ajanın insan müdahalesi olmadan, eğitimine, bağlamına ve tanımlanmış hedeflerine dayanarak seçimler yapma ve eylemler gerçekleştirme yeteneği.

### **Ajan İş Birliği (Agent Collaboration)**
Birden fazla ajanın birlikte çalışma, bilgi paylaşma ve farklı beceriler veya perspektifler gerektiren sorunları çözmek için koordine olma yeteneği.

### **Ajanlar Arası Protokol (A2A - Agent-to-Agent Protocol)**
Google'ın 2025'te tanıttığı açık standart; farklı platformlarda ve uygulamalarda yapay zeka ajanlarının birbirini keşfetmesini, iletişim kurmasını ve iş birliği yapmasını sağlayan protokol.

### **Ajanlar Araç Olarak (Agents as Tools)**
Uzmanlaşmış yapay zeka ajanlarının çağrılabilir fonksiyonlar (araçlar) olarak sarıldığı mimari desen; orkestratör ajanların uzman ajanları koordine ettiği hiyerarşik yapı oluşturur.

### **Sürü Zekası (Swarm Intelligence)**
Birden fazla ajanın paylaşılan bağlam ve çalışma belleğiyle bir ekip olarak çalıştığı iş birlikçi ajan orkestrasyon sistemi; özerk koordinasyon, dinamik görev dağılımı ve kolektif zekayı mümkün kılar.

### **Ajan Sürüsü (Agent Swarm)**
Ajanların kendi kendini organize etme, araç tabanlı koordinasyon ve özerk iş birliği yoluyla ortaya çıkan zeka ile çalıştığı sürü zekasının spesifik uygulaması (OpenAI Swarm kütüphanesi ile popüler olmuştur).

### **Geri Bildirim Döngüsü (Feedback Loop)**
Bir ajanın eylemlerinin sonuçlarından öğrenerek gelecekteki davranışlarını buna göre ayarlamasını sağlayan mekanizma; sürekli iyileşmeyi mümkün kılar.

### **Ajan Durumu (Agent State)**
Bir ajanın belleği, aktif görevleri ve çevresel farkındalığı dahil olmak üzere herhangi bir anda mevcut durumu ve bağlamı.

---

## Model ve Çerçeve Terimleri

### **Büyük Dil Modeli (LLM - Large Language Model)**
GPT-4, Claude, Gemini, Llama, DeepSeek gibi insan benzeri metin anlama ve üretme için geniş metin verileri üzerinde eğitilmiş yapay zeka modeli türü.

### **Temel Model (Foundation Model)**
Çeşitli uygulamalar için temel oluşturan, geniş veri setleri üzerinde önceden eğitilmiş yapay zeka modelleri. Belirli görevler için ince ayar yapılabilir.

### **İstem Mühendisliği (Prompt Engineering)**
Yapay zeka modellerini istenen çıktılara ve davranışlara yönlendirmek için etkili girdi istemleri (prompt) hazırlama uygulaması.

### **Sistem İstemi (System Prompt)**
Bir yapay zeka ajanına verilen, konuşma boyunca rolünü, davranışını ve kısıtlamalarını tanımlayan başlangıç talimatları.

### **Az Örnekli Öğrenme (Few-Shot Learning)**
Yapay zeka modellerinin kapsamlı eğitim verileri yerine sadece birkaç örnek ile görevleri öğrenme tekniği.

### **Akıllı Arama Artırımlı Üretim (RAG - Retrieval-Augmented Generation)**
Ajanların harici bilgi kaynaklarına erişerek daha doğru ve güncel yanıtlar vermesini sağlayan; bilgi erişimi ile metin üretimini birleştiren teknik.

### **Vektör Veritabanı (Vector Database)**
Semantik arama ve RAG uygulamaları için kullanılan; yüksek boyutlu vektörleri depolamak ve sorgulamak için tasarlanmış özel veritabanı.

### **Gömü Vektörleri (Embeddings)**
Anlamsal anlamı yakalayan ve benzerlik karşılaştırmalarını mümkün kılan; metin, görüntü veya diğer verilerin sayısal temsilleri.

### **İnce Ayar (Fine-Tuning)**
Önceden eğitilmiş bir modeli belirli görevler veya alanlar için performansını artırmak üzere belirli verilerle daha da eğitme süreci.

### **Çıkarım (Inference)**
Eğitilmiş bir yapay zeka modelini kullanarak yeni girdi verilerine dayalı tahminler veya çıktılar üretme süreci.

### **Sıfır Örnekli Öğrenme (Zero-Shot Learning)**
Yapay zeka modelinin yalnızca genel bilgi ve anlayışını kullanarak açıkça eğitilmediği görevleri gerçekleştirme yeteneği.

### **Bağlam İçi Öğrenme (In-Context Learning)**
Modelin mevcut konuşma bağlamında sağlanan örnekler veya talimatlara dayanarak yeni görevleri öğrenme ve adapte olma yeteneği.

### **Model Hizalama (Model Alignment)**
Yapay zeka modellerinin insan değerleri ve niyetleriyle tutarlı davranmasını sağlama süreci.

### **İnsan Geri Bildiriminden Pekiştirmeli Öğrenme (RLHF)**
İnsan tercihleri ve geri bildirimlerini kullanarak yapay zeka modeli davranışını ve insan değerleriyle uyumunu iyileştiren eğitim tekniği.

### **Graf Tabanlı Akıl Yürütme (Graph-Based Reasoning)**
Yapay zeka ajanlarının varlıklar, kavramlar veya görevler arasındaki ilişkileri temsil etmek için graf yapıları kullanarak karmaşık ilişkisel akıl yürütme ve bilgi geçişi yapmasını sağlayan yaklaşım.

### **İş Akışı Orkestrasyonu (Workflow Orchestration)**
Belirli iş veya teknik hedeflere ulaşmak için birden fazla ajan, araç ve karar noktasını içeren karmaşık çok adımlı süreçlerin otomatik koordinasyonu ve yönetimi.

### **Ajan Grafiği (Agent Graph)**
Karmaşık çoklu ajan sistemlerini görselleştirmek ve yönetmek için kullanılan; ajanlar ve ilişkileri, bağlantıları ve iletişim yollarının ağ temsili.

### **Hiyerarşik Ajan Mimarisi (Hierarchical Agent Architecture)**
Ajanların katmanlar veya seviyeler halinde organize edildiği; üst düzey ajanların alt düzey ajanları koordine edip yönlendirdiği yapılandırılmış yaklaşım.

### **Ajan Pazarı (Agent Marketplace)**
Farklı yapay zeka ajanlarının keşfedilebileceği, paylaşılabileceği ve entegre edilebileceği platform veya ekosistem; kullanıcıların çeşitli sağlayıcılardan uzmanlaşmış ajanları bulup kullanmasına olanak tanır.

---

## Geliştirme ve Dağıtım

### **SDK (Yazılım Geliştirme Kiti)**
Geliştiricilerin belirli platformlar veya hizmetler için uygulama oluşturmasını sağlayan yazılım geliştirme araçları, kütüphaneler ve dokümantasyon koleksiyonu.

### **API (Uygulama Programlama Arayüzü)**
Farklı yazılım uygulamalarının birbiriyle iletişim kurmasını sağlayan protokol ve araç seti.

### **Sunucusuz (Serverless)**
Bulut sağlayıcısının altyapıyı yönettiği, geliştiricilerin sunucu yönetimi yapmadan koda odaklanmasına olanak tanıyan bulut bilişim modeli.

### **Konteynerleştirme (Containerization)**
Uygulamaların ve bağımlılıklarının farklı ortamlarda tutarlı dağıtım için konteynerlere paketlenmesi uygulaması.

### **Docker**
Konteynerleştirme teknolojisi kullanarak uygulamaları geliştirme, gönderme ve çalıştırma platformu.

### **ALTYAPI KODU (Infrastructure as Code - IaC)**
Altyapının manuel süreçler yerine kod aracılığıyla yönetilmesi ve sağlanması uygulaması.

### **CDK (Bulut Geliştirme Kiti)**
AWS'nin tanıdık programlama dillerini kullanarak bulut altyapısını tanımlama çerçevesi.

---

## Güvenlik ve Uyumluluk

### **KIMLIK ve ERİŞİM YÖNETİMİ (IAM)**
AWS kaynaklarına erişim için kullanıcı kimliklerini ve izinlerini yöneten AWS hizmeti.

### **SIFIR GÜVEN MİMARİSİ (Zero Trust Architecture)**
Konumlarına bakılmaksızın her kullanıcı ve cihazın kaynaklara erişim için doğrulama gerektiren güvenlik modeli.

### **EN AZ AYRICALIK İLKESİ (Principle of Least Privilege)**
Kullanıcı ve sistemlerin işlevlerini yerine getirmek için ihtiyaç duydukları minimum erişim seviyelerinin verildiği güvenlik konsepti.

### **ŞİFRELEME (Encryption)**
Verinin diskte veya veritabanında depolandığında (Rest) ve sistemler arasında hareket ederken (Transit) korunması.

### **DENETİM İZİ (Audit Trail)**
Olayların yeniden yapılandırılmasını ve incelenmesini sağlayan kronolojik sistem aktivitesi kaydı.

---

## Performans ve İzleme

### **GÖZLEMLENEŞEBİLİRLİK (Observability)**
Loglar, metrikler ve izler dahil olmak üzere dış çıktılarına dayanarak bir sistemin iç durumunu anlama yeteneği.

### **TELEMETRİ (Telemetry)**
Uzaktan izleme ve analiz için sistemlerden verilerin otomatik olarak toplanması ve iletilmesi.

### **GECİKMENİ (Latency)**
Bir istek ile yanıtı arasındaki zaman gecikmesi; yapay zeka uygulamalarında kullanıcı deneyimi için kritik.

### **VERİM (Throughput)**
Bir sistemin birim zamanda işleyebileceği istek veya işlem sayısı.

### **OTOMATIK ÖLÇEKLENDİRME (Auto Scaling)**
Talebe bağlı olarak işlem kaynaklarının otomatik olarak ayarlanması; performansı korur ve maliyeti optimize eder.

---

## Entegrasyon ve Protokoller

### **MODEL BAĞLAM PROTOKOLÜ (MCP - Model Context Protocol)**
Anthropic tarafından geliştirilen, yapay zeka ajanlarının çeşitli veri kaynaklarına ve araçlara güvenli bir şekilde bağlanmasını ve etkileşimde bulunmasını sağlayan standart protokol.

### **REST API**
HTTP metodlarını kullanarak sistemler arası iletişim için web servis mimarisi.

### **GraphQL**
İstemcilerin belirli verileri isteyebildiği API için sorgu dili ve çalışma zamanı.

### **WEBHOOK**
Uygulamaların HTTP geri çağrıları göndererek diğer uygulamalara gerçek zamanlı bilgi sağlama yöntemi.

### **OAUTH**
Uygulamaların kullanıcı hesaplarına sınırlı erişim elde etmesini sağlayan yetkilendirme çerçevesi.

### **JSON**
İnsanlar tarafından okunup yazılabilen hafif veri değişim formatı.

### **OLAY TABANLI MİMARİ (Event-Driven Architecture)**
Bileşenlerin olayların üretimi ve tüketimi aracılığıyla iletişim kurduğu yazılım mimarisi modeli.

---

## İş ve Operasyonel Terimler

### **YZ STRATEJİSİ**
Bir kuruluşun yapay zeka teknolojilerini iş hedeflerine ve rekabet avantajlarına nasıl uygulayacağını özetleyen kapsamlı plan.

### **DİJİTAL DÖNÜŞÜM**
Dijital teknolojinin işin tüm alanlarına entegrasyonu; kuruluşların nasıl çalıştığını ve müşterilere değer sunma biçimini temelden değiştirme süreci.

### **YATIRIM GETİRİSİ (ROI)**
Yapay zeka uygulamalarının maliyetlerine göre verimliliğinin ve karlılığının değerlendirildiği performans ölçütü.

### **TOPLAM SAHİPLİK MALİYETİ (TCO)**
Geliştirme, dağıtım, işletme ve bakım masrafları dahil yapay zeka sisteminin uygulanmasının ve sürdürülmesinin tam maliyeti.

### **KAVRAM KANITI (PoC - Proof of Concept)**
Tam uygulamadan önce bir yapay zeka çözümü konseptinin pratik potansiyelini doğrulamak için tasarlanan küçük ölçekli gösterim.

### **EN DÜŞÜK İŞLEVSEL ÜRÜN (MVP)**
Kullanıcı geri bildirimi toplamak ve iş varsayımlarını doğrulamak için piyasaya sürülebilecek en basit yapay zeka ürünü versiyonu.

### **YZ ETİĞİ**
Yapay zeka sistemlerinin geliştirilmesini ve dağıtımını yöneten ahlaki ilkeler ve kılavuzlar; adalet, şeffaflık ve hesap verebilirliği sağlar.

### **YANLIŞLIK AZALTMA (Bias Mitigation)**
Yapay zeka modellerinde ve çıktılarında adil olmayan önyargıları belirlemek, ölçmek ve azaltmak için kullanılan teknikler ve süreçler.

### **AÇIKLANABİLİR YZ (XAI)**
Kararlarını ve önerilerini insan kullanıcılara açık ve anlaşılabilir açıklamalarla sunmak için tasarlanmış yapay zeka sistemleri.

### **İNSAN DÖNGÜSÜNDE (HITL - Human-in-the-Loop)**
İnsanların gözetim, doğrulama veya müdahale sağladığı yapay zeka karar alma sürecinde yer aldığı model.

### **YZ BENİMSEME EĞRİSİ**
Kuruluşların genellikle farkındalık, deney, pilot projeler ve tam dağıtım aşamalarından geçerek yapay zeka teknolojilerini kademeli olarak entegre etme süreci.

### **HİZMET SEVİYESİ ANLAŞMASI (SLA)**
Yapay zeka hizmetleri için beklenen performans standartlarını ve kullanılabilirlik garantilerini tanımlayan sözleşme.

---

## İleri Düzey Kavramlar

### **İSTEM ENJEKSİYONU (Prompt Injection)**
Kötü amaçlı girdilerin yapay zeka modelinin davranışını manipüle etmek veya hassas bilgileri çıkarmak için tasarlandığı güvenlik açığı.

### **HALÜSİNASYON (Hallucination)**
Yapay zeka modelinin makul görünen ancak gerçekte yanlış veya eğitim verilerine dayanmayan bilgiler üretmesi.

### **SICAKLIK (Temperature)**
Yapay zeka modeli çıktılarının rastgeleliğini kontrol eden parametre; daha yüksek değerler daha yaratıcı ancak daha az öngörülebilir yanıtlar üretir.

### **TOKEN**
Yapay zeka modellerinde metin işlemenin temel birimi; kelimeleri, kelime parçalarını veya karakterleri temsil edebilir.

### **TOKENİZASYON (Tokenization)**
Metnin yapay zeka modelleri tarafından işlenebilir tokenlara ayrılması süreci.

### **DİKKAT MEKANİZMASI (Attention Mechanism)**
Modellerin tahminler yaparken girdi verilerinin ilgili bölümlerine odaklanmasını sağlayan sinir ağları tekniği.

### **TRANSFORMER MİMARİSİ**
Dikkat mekanizmalarına dayanan, modern dil modellerinin çoğunda kullanılan temel sinir ağı mimarisi.

### **DÜŞÜNCE ZİNCİRİ (Chain of Thought - CoT)**
Yapay zeka modellerini adım adım akıl yürütme süreçlerini göstermeye teşvik ederek daha doğru ve açıklanabilir sonuçlara yol açan istem tekniği.

### **REAKT (Reasoning and Acting - ReAct)**
Yapay zeka ajanlarında akıl yürütme ve eylemi birleştiren çerçeve; karmaşık sorunları daha etkili çözmek için düşünce süreçleriyle eylemleri birbirine katar.

### **DÜŞÜNCE AĞACI (Tree of Thoughts)**
Olası çözümlerin ağaç benzeri bir yapısını oluşturarak birden fazla akıl yürütme yolunu aynı anda keşfeden gelişmiş akıl yürütme çerçevesi.

### **ANAYASAL YZ (Constitutional AI)**
Yapay zeka sistemlerini daha faydalı, zararsız ve dürüst yapmak için bir ilke veya "anayasa" seti kullanarak eğitme yaklaşımı.

### **UZMAN KARIŞIMI (MoE - Mixture of Experts)**
Girdileri en uygun uzmana yönlendirmek için bir geçiş mekanizması kullanan çoklu uzmanlaşmış alt modeller kullanan sinir ağı mimarisi (örn. DeepSeek V3, GPT-4).

### **AJAN BELLEĞİ (Agent Memory)**
Bir yapay zeka ajanının önceki etkileşimlerden bilgi depolama ve hatırlama yeteneği; oturumlar arasında süreklilik ve öğrenmeyi mümkün kılar.

### **BÖLÜMSEL BELLEK (Episodic Memory)**
Bir konuşma veya görev oturumu içindeki son etkileşimleri ve bağlamı depolayan kısa süreli bellek.

### **ANLAMSAL BELLEK (Semantic Memory)**
Farklı oturumlar ve görevler arasında kalıcı kalan, öğrenilmiş kavramları depolayan uzun süreli bellek.

### **AJAN KİŞİLİĞİ (Agent Persona)**
Bir yapay zeka ajanının kullanıcılarla nasıl etkileşim kurduğunu ve yanıt verdiğini şekillendiren tanımlanmış kişilik, rol ve davranışsal özellikler.

### **AMAÇ YÖNELİMLİ DAVRANIŞ (Goal-Oriented Behavior)**
Ajanın bu hedeflere doğru ilerlemeye bağlı olarak eylemlerini uyarlayarak belirli hedeflere yönelik çalışma yeteneği.

### **ORTAYA ÇIKAN DAVRANIŞ (Emergent Behavior)**
Bir ajanın bileşenlerinin veya bir sistemdeki birden fazla ajanın karmaşık etkileşimlerinden ortaya çıkan beklenmedik veya planlanmamış davranışlar.

### **MODEL KAYMASI (Model Drift)**
Veri desenleri veya çevresel koşullardaki değişiklikler nedeniyle modelin performansının zamanla kademeli olarak bozulması.

### **YZ YÖNETİŞİMİ (AI Governance)**
Yapay zeka sistemlerinin sorumlu geliştirilmesini ve dağıtımını sağlayan politikalar, prosedürler ve kontroller çerçevesi.

### **AKIŞ YANITI (Streaming Response)**
Tam yanıtın tamamlanmasını beklemeden, yapay zeka tarafından üretilen içeriğin gerçek zamanlı olarak iletilmesi.

### **HIZ SINIRLAMA (Rate Limiting)**
Bir kullanıcının veya sistemin belirli bir zaman diliminde yapay zeka hizmetine yapabileceği istek sayısını kısıtlayan kontroller.

### **TOPLU İŞLEME (Batch Processing)**
Birden fazla yapay zeka görevinin bir grup olarak birlikte yürütülmesi; genellikle büyük ölçekli operasyonlar için daha verimli.

### **PLATFORMLAR ARASI ENTEGRASYON (Cross-Platform Integration)**
Standart protokoller ve arayüzler kullanarak farklı platformlarda, çerçevelerde ve uygulamalarda ajanları bağlama ve koordine etme yeteneği.

### **AJAN KEŞFİ (Agent Discovery)**
Ajanların bir ağ veya ekosistemde otomatik olarak diğer mevcut ajanları bulup tanımlaması süreci; dinamik iş birliği ve kaynak kullanımını mümkün kılar.

### **DİNAMİK GÖREV DAĞILIMI (Dynamic Task Distribution)**
Görevlerin ajanlar arasında mevcut yeteneklerine, kullanılabilirliklerine ve iş yüklerine göre otomatik olarak tahsisi ve yeniden ataması; sistem performansını ve verimliliğini optimize eder.

### **KOLLEKTİF ZEKA (Collective Intelligence)**
Birden fazla ajan birlikte çalıştığında ortaya çıkan artırılmış bilişsel yetenekler; karmaşık sorunları çözmek için bireysel bilgi ve işlem güçlerini birleştirir.

### **KENDİNİ DÜZENLEYEN SİSTEMLER (Self-Organizing Systems)**
Merkezi kontrol veya önceden belirlenmiş hiyerarşiler olmadan kendilerini otomatik olarak yapılandırabilen, bağlantılar kurabilen ve organizasyonlarını adapte edebilen ajan ağları.

### **ARAÇ SEÇİM REHBERİ (Tool Selection Guidance)**
Orkestratör ajanların bağlam, gereksinimler ve performans kriterlerine dayanarak belirli görevler için hangi uzmanlaşmış araçları veya ajanları kullanacağına karar vermesine yardımcı olan mekanizmalar.

### **AJAN DEVRİ (Agent Handoff)**
Bir görevin kontrolünün veya sorumluluğunun bir ajandan diğerine devredilmesi süreci; genellikle işin devamı için farklı uzmanlık veya yetenekler gerektiğinde gerçekleşir.

### **MODÜLER MİMARİ (Modular Architecture)**
Ajanların ve bileşenlerinin bağımsız olarak geliştirilebildiği, dağıtılabildiği ve değiştirilebildiği; tüm sistemi etkilemeden tasarım yaklaşımı.

---

## 📊 Hızlı Referans Tablosu

| Terim | Kategori | Seviye | Ana Kullanım Alanı |
|-------|----------|--------|-------------------|
| Ajan | Temel | Başlangıç | Özerk yapay zeka sistemi |
| LLM | Model | Başlangıç | Metin üretimi ve anlama |
| RAG | Çerçeve | Orta | Bilgi artırımlı yanıtlar |
| ReAct | Çerçeve | Orta | Akıl yürütme ve eylem birleşimi |
| MCP | Protokol | İleri | Ajan-veri kaynağı entegrasyonu |
| A2A | Protokol | İleri | Ajanlar arası iletişim |
| Sürü Zekası | Temel | İleri | İş birlikçi ajan orkestrasyonu |
| MCP | Güvenlik | İleri | Model bağlam protokolü |
| İnce Ayar | Model | Orta | Özelleştirilmiş model eğitimi |
| İstem Enjeksiyonu | Güvenlik | Orta | Güvenlik açığı |
| Açıklanabilir YZ | İş | Orta | Hesap verebilirlik |

---

## 💡 Sonuç

Bu terminoloji rehberi, hızla gelişen Ajanik YZ dünyasında pusulanız görevi görüyor. Alan ilerlemeye devam ettikçe yeni kavramlar, çerçeveler ve teknolojiler ortaya çıkacaktır. Yapay zeka yolculuğunuzda yeni terimlerle karşılaştıkça bu rehberi yer imlerine ekleyip düzenli olarak ziyaret etmenizi öneririz.

Unutmayın, bu kavramları anlamak sadece başlangıçtır. Gerçek değer, bunları gerçek dünya sorunlarını çözmek ve anlamlı yapay zeka çözümleri oluşturmak için uygulamaktan gelir. İster ilk sohbet robotunuzu oluşturun, ister kurumsal ölçekli ajan orkestrasyonu uygulayın, ister yapay zeka akıl yürütmesindeki bir sonraki atılımı araştırın, bu terminolojiye sağlam bir kavrayış ilerlemenizi hızlandıracak ve alandaki diğer kişilerle iletişiminizi iyileştirecektir.

Yapay zekanın geleceği ajanik, özerk ve inanılmaz derecede heyecan verici. Bu bilgiyle donanmış olarak, bu geleceğin bir parçası olmaya hazırsınız.

---
www.akkazeta.com.tr
www.ankasofia.com.tr
www.vodoo.com.tr

