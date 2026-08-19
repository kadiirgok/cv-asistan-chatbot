# -*- coding: utf-8 -*-
"""
Daha kapsamlı (stres) test veri seti oluşturan script.

5 adet, her biri tek chunk'a sığmayacak kadar uzun (birden çok chunk'a
bölünen) Türkçe dokümanı mevcut chroma_db'ye EKLER. build_index artık
içerik-hash tabanlı id kullandığı için mevcut 6 test dokümanının üzerine
YAZMAZ, yanına ekler.
"""

from rag import build_index, chunk_text


def main():
    """5 uzun test dokümanını mevcut indekse ekler."""
    belgeler = [
        # 1) Bir şehir hakkında detaylı bilgi (tarih + nüfus + ekonomi + kültür)
        """Bursa, Türkiye'nin kuzeybatısında, Marmara Bölgesi'nin güneyinde ve Uludağ'ın kuzey eteklerinde kurulmuş büyük bir şehirdir. Coğrafi konumu sayesinde hem deniz hem de dağ turizmine elverişli olan şehir, İstanbul'a yakınlığı nedeniyle önemli bir ticaret ve sanayi merkezi haline gelmiştir. Zengin yeşil dokusu ve tarihi yapılarıyla tanınan şehir, halk arasında Yeşil Bursa olarak anılır.

Tarih açısından Bursa, Osmanlı Devleti'nin ilk başkenti olma özelliğini taşır. Şehir, 1326 yılında Orhan Gazi tarafından fethedilmiş ve ardından Osmanlı Beyliği'nin yönetim merkezi olarak kullanılmıştır. Bu dönemden günümüze ulaşan Ulu Cami, Yeşil Türbe ve Koza Han gibi önemli yapılar, erken dönem Osmanlı mimarisinin en güzel örnekleri arasında gösterilir. Bursa, bu tarihi dokusu sayesinde UNESCO Dünya Mirası listesinde de yer almaktadır.

Nüfus bakımından Bursa, yaklaşık üç milyon iki yüz bin kişilik nüfusuyla İstanbul, Ankara ve İzmir'in ardından Türkiye'nin en kalabalık dördüncü şehri konumundadır. Yirminci yüzyılın ikinci yarısından itibaren sanayileşme ve kırsaldan gelen göç, şehir nüfusunun hızla artmasına neden olmuştur. Nüfusun büyük bölümü Osmangazi, Yıldırım ve Nilüfer gibi merkez ilçelerde yaşamaktadır.

Ekonomi alanında Bursa, Türkiye'nin en güçlü sanayi şehirlerinden biridir. Özellikle otomotiv sektörü şehrin ekonomisinin lokomotifi durumundadır; çok sayıda otomobil fabrikası ve yan sanayi tesisi bu şehirde üretim yapmaktadır. Bunun yanında tekstil ve tarihsel olarak ipekçilik de Bursa ekonomisinde köklü bir yere sahiptir; Koza Han bir dönem ipek ticaretinin merkezi olmuştur. Tarım da önemli bir geçim kaynağıdır; şeftali, kestane ve kiraz gibi ürünler şehrin simgesi haline gelmiştir.

Kültür ve gastronomi söz konusu olduğunda Bursa, dünyaca ünlü İskender kebabı ile öne çıkar. Bu yemek, ince dilimlenmiş döner etinin tereyağı, domates sosu ve yoğurtla birlikte servis edilmesiyle hazırlanır. Kestane şekeri de şehrin geleneksel tatlıları arasındadır. Uludağ ise kış turizminin en önemli merkezlerinden biri olarak kayak tutkunlarını ağırlar ve şehre hem turizm geliri hem de doğal güzellik kazandırır. Ayrıca şehirdeki termal kaynaklar kaplıca turizminin gelişmesini sağlamıştır.

Eğitim alanında Bursa, Uludağ Üniversitesi ve Bursa Teknik Üniversitesi gibi köklü yükseköğretim kurumlarına ev sahipliği yapar. Ulaşım açısından şehir, BursaRay hafif raylı sistemi, gelişmiş karayolu ağı ve feribot seferleri sayesinde İstanbul ve bölge illeriyle güçlü bağlantılara sahiptir. Sanayisi, tarihi dokusu ve doğal güzellikleriyle Bursa, hem yaşamak hem de yatırım yapmak için tercih edilen önemli bir merkez konumundadır.""",

        # 2) Bir teknoloji/yazılım kavramının açıklaması
        """Mikroservis mimarisi, bir yazılım uygulamasının birbirinden bağımsız olarak geliştirilebilen, dağıtılabilen ve ölçeklendirilebilen küçük servislerden oluşan bir yapıda tasarlanması yaklaşımıdır. Geleneksel yekpare monolitik mimarinin aksine, mikroservis mimarisinde uygulamanın her bir işlevi ayrı bir servis olarak ele alınır. Örneğin bir e-ticaret uygulamasında ürün kataloğu, sepet, ödeme ve kullanıcı yönetimi ayrı servisler halinde geliştirilebilir.

Bu mimarinin en önemli avantajlarından biri, her servisin kendi teknoloji yığınıyla ve kendi geliştirme ekibi tarafından bağımsız olarak geliştirilip dağıtılabilmesidir. Bir serviste yapılan değişiklik diğer servisleri etkilemeden yayına alınabilir; bu da sürekli teslimat ve hızlı iterasyon imkânı sağlar. Ayrıca yüksek trafik alan belirli bir servis diğerlerinden bağımsız olarak ölçeklendirilebilir, böylece kaynak kullanımı daha verimli hale gelir.

Öte yandan mikroservis mimarisinin bazı dezavantajları da vardır. Dağıtık bir sistem olduğu için servisler arasındaki iletişim ağ gecikmesi ve hata yönetimi gibi ek karmaşıklıklar getirir. Servislerin birbiriyle veri paylaşması ve tutarlılığı sağlaması monolitik yapılara göre daha zordur. Ayrıca çok sayıda servisin izlenmesi, günlüklerinin toplanması ve hata ayıklanması ek altyapı ve araçlar gerektirir.

Mikroservisler genellikle birbirleriyle REST API'leri veya mesaj kuyrukları aracılığıyla iletişim kurar. Her servis kendi veritabanına sahip olabilir ve bu sayede veri modeli üzerinde tam kontrol sahibi olur. Günümüzde mikroservislerin paketlenmesi ve çalıştırılması için Docker gibi konteyner teknolojileri, yönetimi ve ölçeklendirilmesi için ise Kubernetes gibi orkestrasyon platformları yaygın olarak kullanılmaktadır. Bu teknolojiler, her servisin izole bir ortamda çalıştırılmasını ve kolayca kopyalanarak ölçeklendirilmesini mümkün kılar.

Sonuç olarak mikroservis mimarisi, özellikle büyük ve sürekli gelişen uygulamalar için esneklik ve ölçeklenebilirlik sağlarken, küçük projelerde gereksiz karmaşıklık yaratabilir. Bu nedenle hangi mimarinin seçileceğine projenin büyüklüğüne ve ekibin yetkinliğine göre karar verilmelidir.

Günümüzde Netflix, Amazon ve Uber gibi dünya çapındaki büyük teknoloji şirketleri sistemlerini mikroservis mimarisiyle geliştirmektedir. Bu şirketler, yüzlerce hatta binlerce mikroservisi aynı anda çalıştırarak milyonlarca kullanıcıya kesintisiz hizmet verir. Bununla birlikte bu kadar çok servisin yönetilmesi, izleme ve gözlemlenebilirlik araçlarına olan ihtiyacı artırmıştır. Prometheus, Grafana ve dağıtık izleme araçları, hataların tespiti ve performans takibi için yaygın olarak kullanılır. Ayrıca API ağ geçidi ve servis keşfi mekanizmaları, mikroservislerin birbirini bulmasını ve dış dünyaya güvenli şekilde açılmasını sağlar.""",

        # 3) Bir tarihi olay anlatımı
        """Çanakkale Savaşı, Birinci Dünya Savaşı sırasında 1915 yılında Osmanlı Devleti ile İtilaf Devletleri arasında, Çanakkale Boğazı ve Gelibolu Yarımadası'nda gerçekleşen bir dizi muharebeden oluşur. Savaşın temel amacı, İtilaf Devletleri'nin Çanakkale Boğazı'nı geçerek İstanbul'u ele geçirmek ve Rusya'ya deniz yoluyla yardım ulaştırmaktı. Bu stratejik hedef, savaşın seyrini belirleyen en önemli unsurlardan biri olmuştur.

Savaşın ilk aşaması deniz harekâtı olarak başlamıştır. 18 Mart 1915 tarihinde İtilaf donanması, Çanakkale Boğazı'nı geçmek için büyük bir saldırı düzenlemiştir. Ancak Osmanlı mayın gemisi Nusret'in bir gece önce döşediği mayınlar ve kıyı topçusunun etkili atışları sonucunda İtilaf donanması ağır kayıplar vermiştir. Bouvet, Ocean ve Irresistible adlı zırhlılar bu saldırı sırasında batmış ve donanma geri çekilmek zorunda kalmıştır.

Deniz harekâtının başarısız olmasının ardından İtilaf Devletleri, 25 Nisan 1915 tarihinde Gelibolu Yarımadası'na çıkarma yaparak kara harekâtını başlatmıştır. Anzak (Avustralya ve Yeni Zelanda Kolordusu) ve İngiliz kuvvetleri yarımadanın çeşitli noktalarına çıkarma yapmıştır. Buna karşılık Osmanlı ordusu, Albay Mustafa Kemal'in etkili komutası ve askerlerin büyük fedakârlığı sayesinde cepheyi savunmayı başarmıştır. Mustafa Kemal'in Ben size taarruzu değil, ölmeyi emrediyorum sözü bu mücadelenin simgesi haline gelmiştir.

Savaş, aylarca süren siper çatışmalarının ardından İtilaf Devletleri'nin 1915 yılı sonunda ve 1916 başında bölgeden tamamen çekilmesiyle sona ermiştir. Her iki taraf da çok ağır kayıplar vermiştir; yüz binlerce asker bu savaşta hayatını kaybetmiştir. Çanakkale Savaşı, Osmanlı Devleti açısından önemli bir savunma zaferi olarak kabul edilir ve Türk milletinin bağımsızlık mücadelesinde önemli bir dönüm noktası olarak görülür. Her yıl 18 Mart'ta Çanakkale Zaferi anma törenleriyle hatırlanan bu savaş, aynı zamanda Mustafa Kemal Atatürk'ün ulusal bir kahraman olarak öne çıkmasını sağlamıştır.

Çanakkale Savaşı'nın anısı, günümüzde Gelibolu Yarımadası'ndaki Çanakkale Şehitleri Abidesi ve çok sayıda şehitlikle yaşatılmaktadır. Savaş, yalnızca Türkiye için değil, savaşa katılan diğer ülkeler için de derin bir anlam taşır; Avustralya ve Yeni Zelanda'da her yıl 25 Nisan'da Anzak Günü olarak anılır. Savaş sırasında cephede gösterilen insanlık ve karşılıklı saygı örnekleri de tarihe geçmiştir. Çanakkale Zaferi, Osmanlı ordusunun azmi ve fedakârlığının bir göstergesi olarak Türk ulusal bilincinin şekillenmesinde önemli rol oynamış ve Millî Mücadele'nin ruhunu ateşlemiştir.""",

        # 4) Bir bilimsel konu (doğa olayı)
        """Deprem, yer kabuğundaki ani enerji boşalması sonucunda oluşan sismik dalgaların yeryüzünü sarsması olayıdır. Depremlerin büyük çoğunluğu, Dünya'nın dış kabuğunu oluşturan ve levha adı verilen büyük kayaç parçalarının hareket etmesiyle meydana gelir. Levha tektoniği kuramına göre bu levhalar sürekli hareket halindedir ve birbirleriyle çarpışır, birbirinden uzaklaşır veya birbirine sürtünür. Levhaların sınırlarında biriken gerilme enerjisi belirli bir eşiği aştığında kayaçların kırılmasıyla aniden boşalır ve deprem oluşur.

Deprem sırasında açığa çıkan enerji sismik dalgalar halinde yayılır. Birincil yani P dalgaları en hızlı yayılan dalgalardır ve hem katı hem sıvı ortamlardan geçebilir. İkincil yani S dalgaları ise P dalgalarından daha yavaştır ve yalnızca katı ortamlardan geçebilir. Bu dalgaların varış süreleri arasındaki fark, depremin merkezinin konumunu belirlemek için kullanılır. Depremin yeryüzündeki başlangıç noktasına merkez üssü (episantr), yerin içindeki kırılmanın olduğu noktaya ise odak (hiposantr) adı verilir.

Bir depremin büyüklüğünü ölçmek için en yaygın kullanılan ölçek Richter ölçeğidir. Bu ölçek, deprem sırasında açığa çıkan enerjiyi logaritmik bir değerle ifade eder; yani ölçekteki her bir birimlik artış yaklaşık otuz kat daha fazla enerjiye karşılık gelir. Depremin şiddeti ise yeryüzünde yarattığı etkiye göre belirlenir ve Mercalli şiddet ölçeği gibi farklı ölçeklerle değerlendirilir.

Türkiye, dünyanın en aktif deprem kuşaklarından birinde yer alır. Ülkenin kuzeyinde Kuzey Anadolu Fay Hattı, doğusunda ise Doğu Anadolu Fay Hattı uzanır. Bu fay hatları üzerinde tarih boyunca çok sayıda yıkıcı deprem meydana gelmiştir. Bu nedenle Türkiye'de depreme hazırlıklı olmak büyük önem taşır. Binaların deprem yönetmeliğine uygun inşa edilmesi, güçlendirme çalışmaları ve halkın deprem bilincinin artırılması, olası can ve mal kaybını azaltmanın en etkili yollarıdır. Ayrıca her evde bir deprem çantası bulundurulması ve tahliye planlarının önceden yapılması önerilir.

Deprem öncesinde ve sonrasında yapılması gereken hazırlıklar hayati önem taşır. Bir deprem çantasında su, konserve gıda, el feneri, ilk yardım malzemeleri, battaniye ve önemli belgelerin kopyaları bulunmalıdır. Deprem sırasında sağlam bir masa altına girilerek çök-kapan-tutun hareketi uygulanması, düşen cisimlerden korunmanın en etkili yoludur. Ana depremden sonra gelen artçı depremler de hasar görmüş binalarda ek risk oluşturabilir. Kıyı bölgelerinde büyük deniz altı depremleri tsunami oluşturabileceğinden, sarsıntının ardından yüksek yerlere çıkılması önerilir. Erken uyarı sistemleri ise depremin merkezinden uzak yerleşimlere saniyeler önce uyarı ulaştırarak hazırlık süresi kazandırabilir.""",

        # 5) Bir yemek tarifi (mercimek çorbasından farklı)
        """Mantı, Türk mutfağının en sevilen geleneksel yemeklerinden biridir. Küçük kareler halinde kesilmiş hamurun içine kıyma konularak katlanması ve haşlanmasıyla hazırlanır. Özellikle Kayseri yöresiyle özdeşleşmiş olan mantı, günümüzde Türkiye'nin dört bir yanında farklı şekillerde yapılmaktadır. Kayseri mantısının en belirgin özelliği parçalarının oldukça küçük olmasıdır; geleneksel olarak bir kaşığa yaklaşık kırk adet mantı sığdığı söylenir.

Mantı yapımına önce hamur hazırlanarak başlanır. Hamur için un, yumurta, su ve bir miktar tuz yoğrularak kulak memesi kıvamında sert bir hamur elde edilir. Hamur, ince bir yufka haline gelinceye kadar açılır ve küçük karelere kesilir. İç harç için kıyma, ince doğranmış soğan, tuz, karabiber ve isteğe bağlı olarak pul biber karıştırılır. Her karenin ortasına az miktarda iç harç konulur ve karenin uçları ortada birleştirilerek bohça şeklinde kapatılır.

Hazırlanan mantılar kaynar tuzlu suya atılarak yaklaşık on dakika haşlanır. Mantılar piştikten sonra süzülür ve servis tabağına alınır. Servis aşamasında üzerine önce sarımsaklı yoğurt dökülür, ardından tereyağında kızdırılmış domates veya biber salçalı sos gezdirilir. Son olarak kuru nane ve sumak serpilerek mantı sıcak şekilde sofraya getirilir. İsteğe göre üzerine pul biber de eklenebilir.

Mantı, besleyici içeriği ve doyurucu yapısıyla Türk sofralarının vazgeçilmez yemekleri arasında yer alır. Yapımı emek isteyen bu yemek, özellikle kalabalık aile sofralarında ve özel günlerde sıklıkla tercih edilir. Bazı yörelerde mantı fırında da pişirilebilir veya farklı şekillerde katlanabilir, ancak en yaygın biçimi haşlanarak yapılan bohça mantısıdır.

Mantının kökeni, Orta Asya'daki göçebe Türk topluluklarına kadar uzanır; Türklerin Anadolu'ya göçüyle birlikte bu yemek Anadolu mutfağına yerleşmiştir. Zaman içinde her yöre mantıyı kendine özgü biçimde yorumlamıştır; örneğin Sinop mantısı fındık büyüklüğünde yapılırken bazı yörelerde mantı fırında pişirilir. Hamurun kurumaması için açılan yufkanın üzerinin nemli bir bezle örtülmesi gibi pratik ipuçları, mantı yapımını kolaylaştırır. Haşlama suyuna eklenen bir miktar tuz ve yağ, mantıların birbirine yapışmasını önler. Doyurucu ve besleyici bir öğün olan mantı, genellikle yanında turşu veya yeşil salata ile servis edilir.""",
    ]

    print(f"{len(belgeler)} stres test dokümanı hazırlandı:\n")
    for i, doc in enumerate(belgeler, 1):
        kelime = len(doc.split())
        chunk_sayisi = len(chunk_text(doc))
        print(f"  Doküman {i}: {kelime} kelime -> {chunk_sayisi} chunk")

    print("\nİndeksleniyor...\n")
    build_index(belgeler)


if __name__ == "__main__":
    main()
