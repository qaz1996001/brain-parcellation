CREATE DATABASE  IF NOT EXISTS `mes_omi` /*!40100 DEFAULT CHARACTER SET utf8 */;
USE `mes_omi`;
-- MySQL dump 10.13  Distrib 5.7.12, for Win32 (AMD64)
--
-- Host: 127.0.0.1    Database: mes_omi
-- ------------------------------------------------------
-- Server version	5.5.5-10.2.7-MariaDB

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `erp_m_powder_bt`
--

DROP TABLE IF EXISTS `erp_m_powder_bt`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `erp_m_powder_bt` (
  `powder_type` varchar(36) NOT NULL,
  `powder_name` varchar(36) DEFAULT NULL,
  `vendor_id` varchar(36) DEFAULT NULL,
  `vendor_name` varchar(36) DEFAULT NULL,
  `note1` varchar(128) DEFAULT NULL,
  `rec_user` varchar(16) DEFAULT NULL,
  `rec_time` varchar(19) DEFAULT NULL,
  `node_id` varchar(16) DEFAULT NULL,
  `node_time` timestamp(6) NULL DEFAULT current_timestamp(6) ON UPDATE current_timestamp(6)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 AVG_ROW_LENGTH=8192 ROW_FORMAT=DYNAMIC;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `erp_m_powder_bt`
--

LOCK TABLES `erp_m_powder_bt` WRITE;
/*!40000 ALTER TABLE `erp_m_powder_bt` DISABLE KEYS */;
INSERT INTO `erp_m_powder_bt` VALUES ('F1234','F12','CC-A1','Vendor1',NULL,'SYS','2020/03/07 14:50:14',NULL,'2020-03-07 06:50:14.378952'),('A-123','AAA','AAA-11','Vendor2',NULL,'SYS','2020/03/07 14:50:14',NULL,'2020-03-07 06:50:14.470124'),('FF-33','BB','BB2-1','Vendor3',NULL,'SYS','2020/03/07 14:50:14',NULL,'2020-03-07 06:50:14.572221'),('DD-23','CC-DD','CC22','Vendor4',NULL,'SYS','2020/03/07 14:50:14',NULL,'2020-03-07 06:50:14.639443'),('S56-34','FF-SS','SS','Vendor5',NULL,'SYS','2020/03/07 14:50:14',NULL,'2020-03-07 06:50:14.707935'),('S50-12','FF-S2','SS','Vendor6',NULL,'SYS','2020/03/07 14:50:14',NULL,'2020-03-07 06:50:14.774216');
/*!40000 ALTER TABLE `erp_m_powder_bt` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-12  8:16:20
